#pragma once
#include "gram_plan.hpp"
#include <algorithm>
#if defined(__HIPCC__)
#define RNS8_GRAM5_HD __host__ __device__
#else
#define RNS8_GRAM5_HD
#endif
namespace rns8::detail {
constexpr bool gram_five_prime(uint32_t q) {
  if(q<2 || q>251)return false;for(uint32_t d=2;d*d<=q;++d)if(q%d==0)return false;return true;
}
struct GramFivePacket {
  uint32_t prefix=0;uint16_t moduli[20]{},a[20]{},b[20]{};uint8_t certified[20]{};
  int64_t n=0,k=0,h=0,hk=0,rows=0,ld=0;bool transpose=false;
};
struct GramFiveResources {
  rns8_status status=RNS8_INVALID_ARGUMENT;GramFivePacket packet;
  uint64_t panel=0,product=0,output=0,products_offset=0,output_offset=0,total=0;
};
inline GramFiveResources query_gram_five_product(const GramProblem& p,const uint16_t* basis,uint64_t budget,bool gpu,bool matrix) {
  GramFiveResources r;const auto support=query_gram_support(p,UINT64_MAX,gpu,false);
  if(support.status!=RNS8_SUCCESS){r.status=support.status;return r;}
  if(!basis)return r;
  auto& v=r.packet;v.prefix=p.prefix;v.n=support.order;v.k=support.k;v.h=int64_t((uint64_t(v.n)+1)/2);
  v.hk=int64_t(((uint64_t(v.k)+1)/2+1)&~UINT64_C(1));v.rows=p.rows;v.ld=p.source_ld;
  v.transpose=p.orientation==GramOrientation::transpose_times_self;
  uint32_t count=0;
  for(uint32_t plane=0;plane<p.prefix;++plane) {
    const uint32_t q=p.modulus?p.modulus:basis[plane];v.moduli[plane]=uint16_t(q);
    if(q<2 || q>256)return r;
    if(!gram_five_prime(q))continue;
    bool found=false;
    for(uint32_t a=0;a<q && !found;++a)for(uint32_t b=0;b<q;++b)if((a*a+b*b)%q==q-1) {
      v.a[plane]=uint16_t(a);v.b[plane]=uint16_t(b);v.certified[plane]=1;found=true;++count;break;
    }
    if(!found)return r;
  }
  if(!count || (matrix && (!gpu || (p.modulus && p.modulus!=251)))){r.status=RNS8_UNSUPPORTED_BACKEND;return r;}
  AddressDomain panel,product,output;
  if(!checked_host_plane_address(v.h,v.hk,v.hk,p.prefix,1,panel) ||
      !checked_host_plane_address(v.h,v.h,v.h,p.prefix,1,product) ||
      !checked_host_plane_address(v.n,v.n,v.n,p.prefix,1,output)){r.status=RNS8_RANGE_ERROR;return r;}
  if(gpu && (!qualified_i32_gemm_address(v.h,v.h,v.hk,v.hk,v.h,v.h,p.prefix) ||
      (matrix && v.hk>65536))){r.status=RNS8_UNSUPPORTED_BACKEND;return r;}
  const auto limit=uint64_t(PTRDIFF_MAX);r.panel=panel.byte_extent;r.product=product.byte_extent;r.output=output.byte_extent;
  if(r.panel>limit/2 || r.product>(limit-2*r.panel)/5 || r.output>limit-2*r.panel-5*r.product){r.status=RNS8_RANGE_ERROR;return r;}
  r.products_offset=2*r.panel;r.output_offset=2*r.panel+5*r.product;r.total=r.output_offset+r.output;
  r.status=r.total>budget?RNS8_WORKSPACE_TOO_SMALL:RNS8_SUCCESS;return r;
}
RNS8_GRAM5_HD inline int gram_five_center(int value,int q) {
  value%=q;if(value<0)value+=q;return value>=(q+1)/2?value-q:value;
}
RNS8_GRAM5_HD inline int gram_five_load(const int8_t* x,const GramFivePacket& p,uint32_t plane,int64_t row,int64_t col) {
  if(row>=p.n || col>=p.k)return 0;
  return x[(int64_t(plane)*p.rows+(p.transpose?col:row))*p.ld+(p.transpose?row:col)];
}
RNS8_GRAM5_HD inline int gram_five_block(const int8_t* x,const GramFivePacket& p,uint32_t plane,int block,int64_t row,int64_t col) {
  return gram_five_load(x,p,plane,row+(block/2)*p.h,col+(block%2)*p.hk);
}
// Y is block diagonal [[a,b],[-b,a]]. The retained certificate is
// a*a+b*b == -1, hence Y*transpose(Y) == -I in the actual prime channel.
RNS8_GRAM5_HD inline int gram_five_right_y(const int8_t* x,const GramFivePacket& p,uint32_t plane,int block,int64_t row,int64_t col) {
  const int first=gram_five_block(x,p,plane,block,row,col),partner=gram_five_block(x,p,plane,block,row,col^1);
  return int(p.a[plane])*first+(col&1?1:-1)*int(p.b[plane])*partner;
}
RNS8_GRAM5_HD inline int8_t gram_five_transform_value(const int8_t* x,const GramFivePacket& p,uint32_t plane,int term,bool left,int64_t row,int64_t col) {
  if(!p.certified[plane])return 0;
  const int q=p.moduli[plane],a11=gram_five_block(x,p,plane,0,row,col),a12=gram_five_block(x,p,plane,1,row,col),a22=gram_five_block(x,p,plane,3,row,col);
  if(term==0)return int8_t(a11);if(term==1)return int8_t(a12);
  const int y21=gram_five_right_y(x,p,plane,2,row,col),y11=gram_five_right_y(x,p,plane,0,row,col);
  const int s1=gram_five_center(y21-y11,q),s2=gram_five_center(a22-y21,q),s3=gram_five_center(s1-a22,q);
  return int8_t(term==2?(left?a22:gram_five_center(s3+a12,q)):term==3?(left?s1:s2):s3);
}
RNS8_GRAM5_HD inline int gram_five_product_at(const int8_t* products,const GramFivePacket& p,uint32_t plane,int term,int64_t row,int64_t col) {
  return products[(int64_t(term)*p.prefix+plane)*p.h*p.h+row*p.h+col];
}
RNS8_GRAM5_HD inline int8_t gram_five_combine_value(const int8_t* products,const GramFivePacket& p,uint32_t plane,int64_t row,int64_t col) {
  const bool top=row<p.h,left=col<p.h;
  int64_t i=row%p.h,j=col%p.h;
  if(top && !left){i=col-p.h;j=row;}
  int result=gram_five_product_at(products,p,plane,0,i,j);
  if(top && left)result+=gram_five_product_at(products,p,plane,1,i,j);
  else {
    result+=gram_five_product_at(products,p,plane,4,i,j)+gram_five_product_at(products,p,plane,3,i,j);
    result+=!top && !left?gram_five_product_at(products,p,plane,3,j,i):gram_five_product_at(products,p,plane,2,i,j);
  }
  return int8_t(gram_five_center(result,p.moduli[plane]));
}
inline void gram_five_product_cpu(const int8_t* source,int8_t* scratch,const GramFiveResources& r) {
  const auto& p=r.packet;auto* left=scratch;auto* right=scratch+r.panel;
  auto* products=scratch+r.products_offset;auto* stage=scratch+r.output_offset;
  for(int term=0;term<5;++term) {
    for(uint32_t plane=0;plane<p.prefix;++plane)for(int64_t row=0;row<p.h;++row)for(int64_t col=0;col<p.hk;++col) {
      left[(int64_t(plane)*p.h+row)*p.hk+col]=gram_five_transform_value(source,p,plane,term,true,row,col);
      right[(int64_t(plane)*p.hk+col)*p.h+row]=gram_five_transform_value(source,p,plane,term,false,row,col);
    }
    for(uint32_t plane=0;plane<p.prefix;++plane)for(int64_t row=0;row<p.h;++row)for(int64_t col=0;col<p.h;++col) {
      int sum=0;for(int64_t k=0;k<p.hk;++k)sum=(sum+int(left[(int64_t(plane)*p.h+row)*p.hk+k])*right[(int64_t(plane)*p.hk+k)*p.h+col])%p.moduli[plane];
      products[(int64_t(term)*p.prefix+plane)*p.h*p.h+row*p.h+col]=int8_t(gram_five_center(sum,p.moduli[plane]));
    }
  }
  for(uint32_t plane=0;plane<p.prefix;++plane)for(int64_t row=0;row<p.n;++row)for(int64_t col=0;col<p.n;++col) {
    if(p.certified[plane])stage[(int64_t(plane)*p.n+row)*p.n+col]=gram_five_combine_value(products,p,plane,row,col);
    else if(row<=col) {
      int sum=0;for(int64_t k=0;k<p.k;++k)sum=(sum+gram_five_load(source,p,plane,row,k)*gram_five_load(source,p,plane,col,k))%p.moduli[plane];
      const auto value=int8_t(gram_five_center(sum,p.moduli[plane]));stage[(int64_t(plane)*p.n+row)*p.n+col]=value;stage[(int64_t(plane)*p.n+col)*p.n+row]=value;
    }
  }
}
}
#undef RNS8_GRAM5_HD
