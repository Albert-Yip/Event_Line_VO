/*
* ELiSeD++ is based on the original ELiSeD algorithm presented by Braendli/Scaramuzza.
*
* Copyright (C) 2016 Andrea Luca Lampart <lamparta at student dot ethz dot ch> (ETH Zurich)
* For more information see <https://github.com/andrealampart/elised_plus_plus>
*/

#include <stdint.h>


#include "dvs_edge_aug/tic_toc.hpp"

namespace event_mapping{

#define TMP template<typename DATA_TYPE>
#define SCOPE EventBuffer<DATA_TYPE>


  TMP
  uint32_t SCOPE::size_ = EB_SMALL;

  TMP
  uint32_t SCOPE::sizeMask_ = size_-1;

  TMP
  SCOPE::EventBuffer(uint32_t size)
  : index_(0)
  {

    size_ = size;
    sizeMask_ = size_-1;
    buffer_ = new DATA_TYPE[size_];


    // Initialize buffer to 0
    uint32_t i = 0;
    for(;i < size_;++i){
      write(DATA_TYPE());
    }



  }

  TMP
  SCOPE::~EventBuffer(){

    delete[] buffer_;

  }

  TMP
  uint32_t inline SCOPE::getSize(){

    return size_;

  }

  TMP
  uint32_t inline SCOPE::getIndex(){

    return index_;

  }

  TMP
  void inline SCOPE::write(DATA_TYPE data)
  {
    buffer_[index_++ & sizeMask_] = data;
    index_ = index_ & sizeMask_ ;
  }

  TMP
  DATA_TYPE inline SCOPE::readn(uint32_t n)
  {
    return buffer_[(index_ + (~n)) & sizeMask_];
  }

  TMP
  void SCOPE::resizeBuffer(uint32_t bufferSize ){

    DATA_TYPE* tmpBuff = new DATA_TYPE[bufferSize];

    uint32_t i = size_;
    for(; i== 0; --i){
      tmpBuff[i] = buffer_[i];
    }

    delete[] buffer_;
    buffer_ = tmpBuff;
    size_ = bufferSize;
    sizeMask_ = bufferSize-1;


  }

  TMP
  DATA_TYPE SCOPE::getOldest(){

    return buffer_[(index_ + 1) & sizeMask_];
  }



#undef TMP
#undef SCOPE


}
