#pragma once
// Generated from the unmodified Apache-2.0 mathematical artifact.
// gsidebottom/logic commit 938ea3feddc3ebf44eace7a55222d0292a96c1dd
// Git blob f3b618ea06ec620180317e6054b3e901b24a0781
// See third_party/rank23-october-55/{LICENSE,NOTICE}.
// Artifact SHA256: 41fa1529abb0bbee09c73f170e49dd50a4ce59b98c71164b8b3f0fcb883d9cd0
// DAG SHA256: 939da465bb46560cf89de34b49491621ef79a31d7169a83dcfc27d1fa14d2988
// Lowered schedule SHA256: b499f1a3bf4640b5a52f9b842ad3c242ebe5fd307e4fb3a39b780208c4051a8d
constexpr slp::RawStage raw_u() {
  slp::RawStage raw{};
  raw.inputs=9; raw.outputs=23; raw.count=13;
  raw.assignments[0]={9,2,5,true,false};
  raw.assignments[1]={10,0,9,true,false};
  raw.assignments[2]={11,1,10,false,false};
  raw.assignments[3]={12,11,5,true,false};
  raw.assignments[4]={13,11,4,true,false};
  raw.assignments[5]={14,8,12,false,false};
  raw.assignments[6]={15,14,1,true,false};
  raw.assignments[7]={16,13,3,true,false};
  raw.assignments[8]={17,6,16,false,false};
  raw.assignments[9]={18,7,17,false,false};
  raw.assignments[10]={19,14,7,true,false};
  raw.assignments[11]={20,3,6,true,false};
  raw.assignments[12]={21,10,20,true,false};
  raw.results[0]=11;
  raw.results[1]=17;
  raw.results[2]=0;
  raw.results[3]=7;
  raw.results[4]=0;
  raw.results[5]=15;
  raw.results[6]=19;
  raw.results[7]=8;
  raw.results[8]=18;
  raw.results[9]=10;
  raw.results[10]=8;
  raw.results[11]=12;
  raw.results[12]=5;
  raw.results[13]=16;
  raw.results[14]=3;
  raw.results[15]=6;
  raw.results[16]=6;
  raw.results[17]=9;
  raw.results[18]=14;
  raw.results[19]=21;
  raw.results[20]=1;
  raw.results[21]=4;
  raw.results[22]=13;
  return raw;
}
constexpr slp::RawStage raw_v() {
  slp::RawStage raw{};
  raw.inputs=9; raw.outputs=23; raw.count=14;
  raw.assignments[0]={9,2,8,false,false};
  raw.assignments[1]={10,0,6,false,false};
  raw.assignments[2]={11,0,3,true,false};
  raw.assignments[3]={12,0,2,false,false};
  raw.assignments[4]={13,5,8,false,false};
  raw.assignments[5]={14,2,11,false,false};
  raw.assignments[6]={15,11,13,true,false};
  raw.assignments[7]={16,14,5,true,false};
  raw.assignments[8]={17,7,10,false,false};
  raw.assignments[9]={18,1,17,false,false};
  raw.assignments[10]={19,17,15,true,false};
  raw.assignments[11]={20,4,19,false,false};
  raw.assignments[12]={21,19,3,true,false};
  raw.assignments[13]={22,21,5,true,false};
  raw.results[0]=19;
  raw.results[1]=14;
  raw.results[2]=9;
  raw.results[3]=4;
  raw.results[4]=18;
  raw.results[5]=13;
  raw.results[6]=5;
  raw.results[7]=6;
  raw.results[8]=3;
  raw.results[9]=15;
  raw.results[10]=7;
  raw.results[11]=22;
  raw.results[12]=7;
  raw.results[13]=12;
  raw.results[14]=1;
  raw.results[15]=1;
  raw.results[16]=2;
  raw.results[17]=10;
  raw.results[18]=21;
  raw.results[19]=11;
  raw.results[20]=20;
  raw.results[21]=4;
  raw.results[22]=16;
  return raw;
}
constexpr slp::RawStage raw_w() {
  slp::RawStage raw{};
  raw.inputs=23; raw.outputs=9; raw.count=28;
  raw.assignments[0]={23,5,18,true,false};
  raw.assignments[1]={24,23,10,false,false};
  raw.assignments[2]={25,7,24,false,false};
  raw.assignments[3]={26,11,25,false,true};
  raw.assignments[4]={27,16,26,false,true};
  raw.assignments[5]={28,1,27,false,false};
  raw.assignments[6]={29,0,12,true,false};
  raw.assignments[7]={30,29,9,false,false};
  raw.assignments[8]={31,13,11,false,true};
  raw.assignments[9]={32,31,28,false,false};
  raw.assignments[10]={33,17,30,false,false};
  raw.assignments[11]={34,25,33,false,false};
  raw.assignments[12]={35,4,20,false,false};
  raw.assignments[13]={36,35,33,true,false};
  raw.assignments[14]={37,2,26,false,false};
  raw.assignments[15]={38,30,19,true,false};
  raw.assignments[16]={39,38,32,false,false};
  raw.assignments[17]={40,14,21,false,false};
  raw.assignments[18]={41,40,12,false,false};
  raw.assignments[19]={42,22,28,true,false};
  raw.assignments[20]={43,42,9,true,false};
  raw.assignments[21]={44,43,19,false,false};
  raw.assignments[22]={45,8,24,true,false};
  raw.assignments[23]={46,45,32,false,false};
  raw.assignments[24]={47,3,15,false,false};
  raw.assignments[25]={48,47,10,false,false};
  raw.assignments[26]={49,5,6,true,false};
  raw.assignments[27]={50,49,27,true,false};
  raw.results[0]=34;
  raw.results[1]=36;
  raw.results[2]=37;
  raw.results[3]=39;
  raw.results[4]=41;
  raw.results[5]=44;
  raw.results[6]=46;
  raw.results[7]=48;
  raw.results[8]=50;
  return raw;
}
