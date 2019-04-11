/*
* ELiSeD++ is based on the original ELiSeD algorithm presented by Braendli/Scaramuzza.
*
* Copyright (C) 2016 Andrea Luca Lampart <lamparta at student dot ethz dot ch> (ETH Zurich)
* For more information see <https://github.com/andrealampart/elised_plus_plus>
*/

#pragma once

#include <stdint.h>
#include <stdlib.h>
#include "dvs_edge_aug/eventbuffer.hpp"
#include <opencv2/core/core.hpp>
#include <map>
#include <set>
#include <list>
#include <unordered_map>
#include <eigen3/Eigen/Core>
#include <mutex>



namespace event_mapping{



// Forward declaration of LineLevelPixel
struct LineLevelPixel;

// Forward declaration of Coin
struct Coin;

/*!
 * \brief The LineSupport class
 */
class LineSupport{

public:

  /*!
   * \brief LineSupport
   * \param pixelMap
   */
  LineSupport( Eigen::Array<LineLevelPixel*, 128, 128> *pixelMap, std::list<LineSupport*> *lines);
  ~LineSupport();

  /*!
   * \brief assignMember
   * \param x
   * \param y
   */
  void assignMember(const uint16_t &x, const uint16_t &y);

  /*!
   * \brief removeMember
   * \param x
   * \param y
   */
  void removeMember(const uint16_t &x, const uint16_t &y);

  /*!
   * \brief getMembers
   */
  std::list<LineLevelPixel*>& getMembers();

  /*!
   * \brief merge
   * \param ls  other lineSupport
   */
  void merge(LineSupport* ls);

  /*!
   * \brief getColor
   */
  cv::Point3f getColor();

  /*!
   * \brief getId
   */
  uint64_t getId();

  /*!
   * \brief getSize
   */
  uint32_t getSize();

  /*!
   * \brief recalculateAngle
   */
  void recalculateAngle();

  /*!
   * \brief getDistanceTo
   * \param ls
   * \return
   */
  float_t getDistanceTo(LineSupport* ls);

  /*!
   * \brief getCenterX
   * \return
   */
  float_t getCenterX();

  /*!
   * \brief getCenterY
   * \return
   */
  float_t getCenterY();

  /*!
   * \brief getAngle
   * \return
   */
  float_t getAngle();

  uint64_t getM00();

  /*!
   * \brief getM01
   * \return
   */
  uint64_t getM01();

  /*!
   * \brief getM10
   * \return
   */
  uint64_t getM10();

  /*!
   * \brief getM11
   * \return
   */
  uint64_t getM11();

  /*!
   * \brief getM20
   * \return
   */
  uint64_t getM20();

  /*!
   * \brief getM02
   * \return
   */
  uint64_t getM02();

  float_t major_axis_,minor_axis_;

  void setAge(uint64_t age);

  uint64_t getAge();

  /*!
   * \brief calculateLine
   */
  void calculateLine();

  /*!
   * \brief getFirst
   * \return
   */
  cv::Point getFirst();

  /*!
   * \brief getSecond
   * \return
   */
  cv::Point getSecond();

  /*!
   * \brief getCentroid
   * \return
   */
  cv::Point2f getCentroid();

  /*!
   * \brief getDirection
   * \return
   */
  cv::Point2f getDirection();

  Coin*& trackCoin();

  bool hasCoin();

  void setCoin(Coin*& coin);

  void handoverCoin(LineSupport* ls);

  static uint64_t idGenerator_;



protected:

  Coin* coinptr_;

  std::list<LineSupport*> *lines_;

  Eigen::Array<LineLevelPixel* , 128, 128> *pixelMap_;

  Eigen::Array<float_t, 128, 128> *angles_;

  uint64_t id_;

  std::mutex ls_mutex_;



  uint64_t age_;

   uint32_t size_;

  std::list<LineLevelPixel*> members_;

  cv::Point3f color_;

  cv::Point2f first_;

  cv::Point2f second_;

  cv::Point2f direction_;

  uint32_t m00, m01, m10, m11, m20, m02;

  float_t x_center_, y_center_;

  float_t length_;

  float_t orientation_;

  float_t angle_;
};

struct LineLevelPixel{

  LineLevelPixel(LineSupport* ls, float_t angle, uint16_t xc, uint16_t yc){
    lineSupport_=ls;
    angle_ = angle;
    x=xc;
    y=yc;
  }

  LineSupport* lineSupport_;


  float_t angle_;
  uint16_t x;
  uint16_t y;
};

/*!
 * \brief The Coin is a way to track a cluster at will. We hand over the coin at merges to the remaining cluster
 */
struct Coin{
  LineSupport* place_;
  Coin** owner_;
  LineSupport* getPlace(){
    return place_;
  }
  void setPlace(LineSupport* place){
    place_ = place;
  }



};



}

