#include "core/distributed_process.hpp"
#include "core/hip_resources.hpp"
#include "backend_common/sha256.hpp"
#define OMPI_SKIP_MPICXX 1
#define MPICH_SKIP_MPICXX 1
#include <mpi.h>
#include <nlohmann/json.hpp>
#include <array>
#include <cstring>
#include <iostream>
#include <sstream>

// Explicit qualification executable, never an automatically launched GPU test.
// Setup collectives/runtime initialization require the bounded external runner.
namespace d=rns8::detail::distributed;
using json=nlohmann::json;
int main(int argc,char** argv) {
  int provided=0;if(MPI_Init_thread(&argc,&argv,MPI_THREAD_SERIALIZED,&provided)!=MPI_SUCCESS)return 1;
  MPI_Comm world=MPI_COMM_WORLD;MPI_Comm_set_errhandler(world,MPI_ERRORS_RETURN);
  int rank=0,count=0;MPI_Comm_rank(world,&rank);MPI_Comm_size(world,&count);
  try {
    if(argc!=4 || count<1 || count>64)throw std::invalid_argument("usage: mpiexec -np N rns8-process-smoke mpi-host|rccl|rocshmem modulus|rows|columns|contraction|hybrid comma-separated-rank-local-ordinals");
    const std::string transport=argv[1],axis=argv[2];
    const auto selected=transport=="mpi-host"?d::Transport::mpi_host:transport=="rccl"?d::Transport::rccl:transport=="rocshmem"?d::Transport::rocshmem:d::Transport::local;
    if(!d::ProcessSession::compiled(selected)){std::cerr<<"selected transport not compiled\n";return 77;}
    std::vector<uint32_t> ordinals;std::istringstream devices(argv[3]);std::string item;
    while(std::getline(devices,item,',')){size_t used=0;auto value=std::stoul(item,&used);if(used!=item.size() || value>INT32_MAX)throw std::invalid_argument("ordinal");ordinals.push_back(uint32_t(value));}
    if(ordinals.size()!=size_t(count))throw std::invalid_argument("one local ordinal per MPI rank required");
    const auto require=[](d::Admission value){if(!value)throw std::runtime_error(std::string(value.reason)+" status="+std::to_string(value.status));};
    d::Rank local;require(d::ProcessSession::discover(&world,ordinals[size_t(rank)],local));
    rns8::detail::hip_scoped_device device(int(local.ordinal));if(device.status()!=hipSuccess)throw std::runtime_error("local HIP selection");
    const auto local_json=json({{"ordinal",local.ordinal},{"node",local.node_id},{"physical",local.physical_id},{"target",local.target},{"memory",local.memory_budget}}).dump();
    if(local_json.size()>=1024)throw std::runtime_error("rank manifest size");
    std::array<char,1024> manifest{};std::copy(local_json.begin(),local_json.end(),manifest.begin());std::vector<char> manifests(size_t(count)*1024);
    if(MPI_Allgather(manifest.data(),1024,MPI_BYTE,manifests.data(),1024,MPI_BYTE,world)!=MPI_SUCCESS)throw std::runtime_error("rank inventory");
    d::Request request;request.operation=51001;request.m=17;request.n=19;request.k=129;request.lda=131;request.ldb=22;
    request.prefix=20;request.tile_rows=8;request.tile_cols=8;request.transport=selected;request.host_storage=d::HostStorage::limbs_only;
    request.host_budget=uint64_t(1)<<30;request.pinned_budget=uint64_t(1)<<20;request.a={71,1};request.b={91,1};
    request.certificate_sha256=rns8::detail::sha256("process-smoke-direct-HIP-safe-K65536");request.layout_sha256=rns8::detail::sha256("native64-row-major-centered-byte-gather-v1");
    request.magnitude_bound=d::Integer(1)<<140;request.native_integrity=true;
    if(axis=="rows")request.axis=d::Axis::rows;else if(axis=="columns")request.axis=d::Axis::columns;
    else if(axis=="contraction"){request.axis=d::Axis::contraction;request.k_fragments=uint32_t(count);}
    else if(axis=="hybrid"){request.axis=d::Axis::hybrid;request.rank_grid_rows=1;request.rank_grid_cols=uint32_t(count);}
    else if(axis!="modulus")throw std::invalid_argument("axis");
    for(int index=0;index<count;++index){const auto entry=json::parse(manifests.data()+size_t(index)*1024);d::Rank value;
      value.ordinal=entry.at("ordinal");value.node_id=entry.at("node");value.physical_id=entry.at("physical");value.target=entry.at("target");value.memory_budget=entry.at("memory");request.ranks.push_back(std::move(value));
      if(index){d::Edge edge;edge.source=uint32_t(index);edge.destination=0;edge.mpi_host=edge.rccl=edge.rocshmem=d::Support::supported;request.edges.push_back(edge);}}
    d::ProcessBinding binding;binding.mpi_communicator=&world;binding.setup_timeout_ns=uint64_t(60)*1000000000;
    binding.acknowledge_unknown_transport_overhead=true;binding.transport_headroom_bytes=selected==d::Transport::mpi_host?0:uint64_t(256)<<20;
    bool device_runtime=false;void* arena=nullptr;
    if(selected==d::Transport::rocshmem){require(d::initialize_device_transport_runtime());device_runtime=true;}
    for(bool signed_values:{false,true}) {
      request.semantics=signed_values?RNS8_EXACT_WIDE_SIGNED:RNS8_EXACT_WIDE_UNSIGNED;
      std::shared_ptr<const d::Plan> plan;require(d::Plan::create(request,plan));
      const uint64_t values[]{UINT64_MAX,uint64_t(1)<<63,0,1,UINT64_MAX-1,0x8080ff007fff0101ull};
      std::vector<uint64_t> a(size_t((request.m-1)*request.lda+request.k),99),b(size_t((request.k-1)*request.ldb+request.n),99);
      for(uint64_t row=0;row<request.m;++row)for(uint64_t k=0;k<request.k;++k)a[size_t(row*request.lda+k)]=values[(row*7+k*3)%6];
      for(uint64_t k=0;k<request.k;++k)for(uint64_t col=0;col<request.n;++col)b[size_t(k*request.ldb+col)]=values[(k*3+col*5)%6];
      auto inputs=std::make_shared<const d::Inputs>(request.a,request.b,a,b);
      d::ProcessResources resources;require(d::ProcessSession::resources(*plan,uint32_t(rank),binding,resources));
      if(selected==d::Transport::rocshmem){require(d::allocate_device_transport_arena(resources.symmetric_arena_bytes,arena));binding.symmetric_arena=arena;binding.symmetric_arena_bytes=resources.symmetric_arena_bytes;}
      std::unique_ptr<d::ProcessSession> session;const auto created=d::ProcessSession::create(plan,inputs,binding,session);require(created);
      for(uint64_t epoch:{1,2}) {
        require(session->execute(epoch,uint64_t(60)*1000000000));
        if(rank==0) {
          require(session->retain(epoch));int ordinal=-1;const void* resident=nullptr;require(session->resident_limbs(epoch,ordinal,resident));if(!resident || ordinal!=int(local.ordinal))throw std::runtime_error("resident owner");
          if(session->execute(epoch+1,1000).status!=RNS8_BUSY)throw std::runtime_error("retained result must reject reuse");
          require(session->release(epoch));
          std::vector<uint64_t> output(size_t(request.m*(request.n+1)*3),0xa5);
          require(session->export_limbs(epoch,output.data(),request.n+1));
          for(uint64_t row=0;row<request.m;++row)for(uint64_t col=0;col<request.n;++col) {
            d::Integer expected=0;for(uint64_t k=0;k<request.k;++k)expected+=d::native_integer(a[size_t(row*request.lda+k)],signed_values)*d::native_integer(b[size_t(k*request.ldb+col)],signed_values);
            const size_t cell=size_t((row*(request.n+1)+col)*3);d::Integer actual=output[cell]+(d::Integer(output[cell+1])<<64)+(d::Integer(output[cell+2])<<128);
            if(signed_values && output[cell+2]>>63)actual-=d::Integer(1)<<192;
            if(actual!=expected)throw std::runtime_error("complete Boost product mismatch");
          }
          for(uint64_t row=0;row<request.m;++row)for(unsigned limb=0;limb<3;++limb)if(output[size_t((row*(request.n+1)+request.n)*3+limb)]!=0xa5)throw std::runtime_error("padding changed");
        }
        // Keep the explicit next-epoch order identical while rank0 verifies.
        if(MPI_Barrier(world)!=MPI_SUCCESS)throw std::runtime_error("qualification epoch barrier");
      }
      require(session->close());session.reset();
      if(arena){require(d::release_device_transport_arena(arena));arena=nullptr;binding.symmetric_arena=nullptr;}
      if(rank==0)std::cout<<json({{"schema","rns8_process_transport_smoke_v1"},{"transport",d::transport_label(selected)},{"axis",axis},{"ranks",count},{"signed_input",signed_values},{"epochs",2},{"complete_boost_reference",true},{"hardware_executed",true},{"production_eligible",false},{"m",request.m},{"n",request.n},{"k",request.k},{"build_manifest",json::parse(RNS8_BUILD_MANIFEST_JSON)}}).dump()<<'\n';
    }
    if(device_runtime)require(d::finalize_device_transport_runtime());
    MPI_Finalize();return 0;
  }catch(const std::exception& error){std::cerr<<"rank "<<rank<<": "<<error.what()<<"; failed owners/arena remain retained until process teardown\n";return 1;}
}
