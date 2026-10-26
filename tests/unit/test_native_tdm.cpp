#include <catch2/catch_test_macros.hpp>
#include "backend_common/native_tdm.hpp"
#include <set>
TEST_CASE("TDM descriptor units bounds and physical two-stage addresses are explicit", "[tdm][cpu]") {
  using namespace rns8::detail;TdmDescriptor2D d;
  REQUIRE(make_native_tdm_descriptor(0x100000000ull,8192,17,65,16,32,67,3,d));
  CHECK(d.group0[0]==1);CHECK(d.group0[1]==8192);CHECK(d.group0[2]==0);CHECK(d.group0[3]==((2u<<30)|1u));
  CHECK(d.group1[0]==((3u<<16)|3));CHECK(d.group1[5]==67);CHECK(d.group1[6]==0);
  CHECK(((d.group1[1]>>16)|(d.group1[2]<<16))==65);
  CHECK(((d.group1[2]>>16)|(d.group1[3]<<16))==17);
  CHECK(d.group1[3]>>16==32);CHECK(d.group1[4]==16);
  for(unsigned rows:{1u,15u,16u})for(unsigned cols:{1u,31u,32u}) {
    for(unsigned row=0;row<16;++row)for(unsigned col=0;col<32;++col) {
      uint64_t source=UINT64_MAX;const bool valid=native_tdm_source(row,col,rows,cols,67,source);
      CHECK(valid==(row<rows && col<cols));if(valid)CHECK(source==row*67+col);
    }
  }
  CHECK_FALSE(make_native_tdm_descriptor(uint64_t{1}<<57,0,1,1,16,32,1,0,d));
  CHECK_FALSE(make_native_tdm_descriptor(8,65528,1,1,16,32,1,0,d));
  CHECK_FALSE(make_native_tdm_descriptor(8,0,1,2,16,32,1,0,d));
  CHECK_FALSE(make_native_tdm_descriptor(8,0,1,1,16,32,1,16,d));
}
TEST_CASE("TDM clustered prefix ownership covers partial groups exactly once", "[tdm][cpu]") {
  using namespace rns8::detail;
  for(unsigned cluster:{1u,2u,4u})for(unsigned group:{1u,2u,4u})for(unsigned prefix=1;prefix<=20;++prefix) {
    NativeTdmProblem p;p.m=17;p.n=19;p.k=33;p.lda=35;p.ldb=21;p.ldc=23;p.prefix=prefix;p.group=group;p.cluster=cluster;p.mode=cluster>1?2:1;
    const auto resources=query_native_tdm(p);REQUIRE(resources.status==RNS8_SUCCESS);CHECK(resources.lds_bytes==32768);
    std::set<uint64_t> cells;const unsigned batches=((prefix+group-1)/group+cluster-1)/cluster;
    for(unsigned z=0;z<batches;++z)for(unsigned x=0;x<2*cluster;++x)for(unsigned y=0;y<2;++y)
      for(unsigned plane=0;plane<group;++plane)for(unsigned thread=0;thread<256;++thread) {
        const unsigned pindex=(z*cluster+x%cluster)*group+plane,row=y*16+thread/16,col=(x/cluster)*16+thread%16;
        if(pindex<prefix && row<17 && col<19)CHECK(cells.insert((uint64_t(pindex)*17+row)*19+col).second);
      }
    CHECK(cells.size()==prefix*17*19);
    p.mode=0;if(cluster>1)CHECK(query_native_tdm(p).status==RNS8_INVALID_ARGUMENT);
  }
}

TEST_CASE("CDNA5 wave32 IU8 model has complete unique fragment coordinates", "[tdm][cpu]") {
  using L=rns8::detail::TdmWmmaLayout;std::set<unsigned> a,b,c;
  for(unsigned lane=0;lane<32;++lane){for(unsigned element=0;element<32;++element){const auto x=L::a(lane,element),y=L::b(lane,element);
    CHECK(x.row<16);CHECK(x.col<64);CHECK(y.row<64);CHECK(y.col<16);CHECK(a.insert(x.row*64+x.col).second);CHECK(b.insert(y.row*16+y.col).second);}
    for(unsigned reg=0;reg<8;++reg){const auto x=L::c(lane,reg);CHECK(x.row<16);CHECK(x.col<16);CHECK(c.insert(x.row*16+x.col).second);}}
  CHECK(a.size()==1024);CHECK(b.size()==1024);CHECK(c.size()==256);
}
TEST_CASE("neighbor multicast padding preserves complete cell ownership", "[tdm][cpu]") {
  using namespace rns8::detail;
  for(unsigned cluster:{2u,4u})for(unsigned prefix:{1u,9u,20u})for(unsigned group:{1u,2u,4u}) {
    NativeTdmProblem p;p.m=17;p.n=19;p.k=65;p.lda=67;p.ldb=21;p.ldc=23;p.prefix=prefix;p.group=group;p.cluster=cluster;p.mode=2;p.cluster_axis=1;p.compute=1;
    const auto resources=query_native_tdm(p);REQUIRE(resources.status==RNS8_SUCCESS);std::set<uint64_t> cells;
    const unsigned columns=((2+cluster-1)/cluster)*cluster,groups=(prefix+group-1)/group;
    CHECK(resources.grid_blocks==2*columns*groups);
    for(unsigned z=0;z<groups;++z)for(unsigned x=0;x<columns;++x)for(unsigned y=0;y<2;++y)for(unsigned channel=0;channel<group;++channel)
      for(unsigned lane=0;lane<32;++lane)for(unsigned reg=0;reg<8;++reg){const auto cell=TdmWmmaLayout::c(lane,reg);const unsigned row=y*16+cell.row,col=x*16+cell.col,plane=z*group+channel;
        if(row<17 && col<19 && plane<prefix)CHECK(cells.insert((uint64_t(plane)*17+row)*19+col).second);}
    CHECK(cells.size()==prefix*17*19);
  }
}
