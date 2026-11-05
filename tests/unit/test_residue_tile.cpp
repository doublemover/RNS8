#include <catch2/catch_test_macros.hpp>
#include <boost/multiprecision/cpp_int.hpp>
#include <algorithm>
#include <vector>
#include "backend_common/residue_tile.hpp"

namespace {
using namespace rns8::detail;
using boost::multiprecision::cpp_int;
int8_t independent_center(cpp_int value,uint32_t q){
  value%=q;if(value<0)value+=q;
  if(value>=(q+1)/2)value-=q;
  return value.convert_to<int8_t>();
}
template<bool Native,uint32_t Fixed>
auto tile_input(const std::vector<uint8_t>& raw,const std::vector<int8_t>& centered,int stride,uint32_t q,std::size_t offset){
  if constexpr(Native)return NativeByteResidueTileInput<Fixed>{raw.data()+offset,stride,q};
  else return CenteredResidueTileInput{centered.data()+offset,stride};
}
template<bool NativeA,bool NativeB,uint32_t Fixed,ResidueTileDot Dot>
void check_tile_policy(const char* name,bool region,int identity){
  INFO(name);
  constexpr int rows=19,cols=21,offset_k=3,block_k=65,total_k=offset_k+block_k;
  constexpr int lda=total_k+5,ldb=cols+7;
  const std::size_t offset=7+identity*13; // distinct task/plane rebases
  for(uint32_t q:{uint32_t(Fixed?Fixed:247),uint32_t(Fixed?Fixed:256)}){
    const std::size_t asize=offset+(rows-1)*lda+total_k+1,bsize=offset+(total_k-1)*ldb+cols+1;
    std::vector<uint8_t> ar(asize),br(bsize);
    std::vector<int8_t> ac(asize),bc(bsize);
    for(std::size_t i=0;i<asize;++i){ar[i]=uint8_t((i*37+identity*19)&255);ac[i]=independent_center(ar[i],q);}
    for(std::size_t i=0;i<bsize;++i){br[i]=uint8_t((i*23+identity*11)&255);bc[i]=independent_center(br[i],q);}
    const auto a=tile_input<NativeA,Fixed>(ar,ac,lda,q,offset);
    const auto b=tile_input<NativeB,Fixed>(br,bc,ldb,q,offset);
    for(int threads:{64,128,256})for(int origin:{0,16}){
      const int tile_row=origin+(region?1:0),tile_col=origin+(region?3:0);
      int32_t accumulators[16][16]{};
      for(int tile_k=0;tile_k<block_k;tile_k+=64){
        int8_t at[16][64],bt[64][17];
        for(auto& row:at)std::fill_n(row,64,int8_t(99));
        for(auto& row:bt)std::fill_n(row,17,int8_t(99));
        const int extent=std::min(64,block_k-tile_k);
        for(int lane=0;lane<threads;++lane)
          load_residue_tile<16,16,64,17>(at,bt,{rows,cols,tile_row,tile_col,offset_k},tile_k,extent,lane,threads,a,b);
        for(int row=0;row<16;++row)for(int x=0;x<64;++x){
          const int global_row=tile_row+row;
          const auto expected=global_row<rows && x<extent?
              independent_center(ar[offset+global_row*lda+offset_k+tile_k+x],q):int8_t(0);
          REQUIRE(at[row][x]==expected);
        }
        for(int x=0;x<64;++x){
          CHECK(bt[x][16]==99); // padded LDS column is not consumed or loaded
          for(int col=0;col<16;++col){
            const int global_col=tile_col+col;
            const auto expected=x<extent && global_col<cols?
                independent_center(br[offset+(offset_k+tile_k+x)*ldb+global_col],q):int8_t(0);
            REQUIRE(bt[x][col]==expected);
          }
        }
        for(int row=0;row<16 && tile_row+row<rows;++row)for(int col=0;col<16 && tile_col+col<cols;++col)
          accumulators[row][col]=accumulate_residue_tile(&at[row][0],&bt[0][col],17,extent,accumulators[row][col],residue_tile_packed<Dot>(int(q)));
      }
      const int8_t carry=independent_center(129,q);
      for(int row=0;row<16 && tile_row+row<rows;++row)for(int col=0;col<16 && tile_col+col<cols;++col){
        cpp_int exact=carry;
        for(int x=0;x<block_k;++x)
          exact+=cpp_int(ar[offset+(tile_row+row)*lda+offset_k+x])*br[offset+(offset_k+x)*ldb+tile_col+col];
        CHECK(finite_u8::reduce_to_centered_i32(accumulators[row][col]+carry,q,finite_u8::modulus_reciprocal_u32(q))==independent_center(exact,q));
      }
    }
  }
}
}
TEST_CASE("thirteen tiled wrapper policy tuples share exact cooperative byte dataflow", "[residue-tile][contract]"){
  using D=ResidueTileDot;
  check_tile_policy<false,false,0,D::configured>("dynamic resident",false,0);
  check_tile_policy<false,false,251,D::configured>("fixed resident",false,1);
  check_tile_policy<false,false,251,D::packed>("fixed packed dot",false,2);
  check_tile_policy<false,false,0,D::configured>("grouped dynamic",false,3);
  check_tile_policy<false,false,251,D::configured>("grouped fixed",false,4);
  check_tile_policy<true,true,0,D::configured>("native AB dynamic",false,5);
  check_tile_policy<true,true,251,D::configured>("native AB fixed",false,6);
  check_tile_policy<true,false,0,D::configured>("native A dynamic",false,7);
  check_tile_policy<true,false,251,D::configured>("native A fixed",false,8);
  check_tile_policy<false,false,0,D::selected_default_moduli>("ordinary default ladder",false,9);
  check_tile_policy<false,false,0,D::configured>("dynamic region",true,10);
  check_tile_policy<false,false,0,D::configured>("default ladder region",true,11);
  check_tile_policy<false,false,0,D::configured>("task prefix",false,12);
  CHECK(residue_tile_packed<D::selected_default_moduli>(255));
  CHECK(residue_tile_packed<D::selected_default_moduli>(251));
  CHECK(residue_tile_packed<D::selected_default_moduli>(247)==kResidueDot4Enabled);
  CHECK(residue_tile_packed<D::configured>(251)==kResidueDot4Enabled);
}
