/*
* ELiSeD++ is based on the original ELiSeD algorithm presented by Braendli/Scaramuzza.
*
* Copyright (C) 2016 Andrea Luca Lampart <lamparta at student dot ethz dot ch> (ETH Zurich)
* For more information see <https://github.com/andrealampart/elised_plus_plus>
*/

#pragma once

#include <stdint.h>



namespace event_mapping{
#define EB_SMALLER 2048
#define EB_SMALL 4096
#define EB_LARGE 8192
#define EB_HUGE (8192*2)
#define EB_YOURMOM 4294967296 // EB_YOURMOM should not be used, as memory allocation can't be handled. EB_YOURMOM is huge.
#define EB_TINY 8


/*
 * Templated class for creating a variable size ringbuffer
 */

template <typename DATA_TYPE>
class EventBuffer{

public:

  EventBuffer( uint32_t size = EB_SMALL);
  ~EventBuffer();
  /*
   * Sets the size of the ring buffer
   * @param bufferSize Size of the buffer
   */
  void resizeBuffer(uint32_t bufferSize);

  /*
   * Get buffer size
   */
  uint32_t inline getSize();

  /*
   * Get current index position
   */
  uint32_t inline getIndex();

  /*
   * Get oldest element in buffer
   */
  DATA_TYPE getOldest();

  /*
   * Inserts new data on top of buffer
   */
  void inline write(DATA_TYPE data);


  /*
   * Read the buffer at place n
   * @param
   */
  DATA_TYPE inline readn(uint32_t n);

protected:

  // Size of event buffer (is identical on all buffer objects)
  static uint32_t size_;

  // Mask for faster processing through binary operations
  static uint32_t sizeMask_;

  // Dynamic size buffer
  DATA_TYPE *buffer_;

  // Index of active element in buffer
  uint32_t index_;



};

}
#include "dvs_edge_aug/eventbuffer.inl"
