/*
* ELiSeD++ is based on the original ELiSeD algorithm presented by
*Braendli/Scaramuzza.
*
* Copyright (C) 2016 Andrea Luca Lampart <lamparta at student dot ethz dot ch>
*(ETH Zurich)
* For more information see <https://github.com/andrealampart/elised_plus_plus>
*/

#include "dvs_edge_aug/line_support.hpp"

#define MAX_ERROR 0.09

namespace event_mapping
{
#define PI 3.14159265359
#define INV_PI 0.31830988618
#define SQRT_SIX 2.44948974278f
#define INVALID_ANGLE 5.0
#define SCOPE LineSupport

uint64_t SCOPE::idGenerator_ = 0;

SCOPE::LineSupport(Eigen::Array<LineLevelPixel*, ROWS_, COLS_>* pixelMap,
                   std::list<LineSupport*>* lines)
    : size_(0), x_center_(0.0f), y_center_(0.0f), m00(0), m01(0), m10(0),
      m11(0), m20(0), m02(0), angle_(INVALID_ANGLE),coinptr_(nullptr)
{

  idGenerator_++;
  id_ = idGenerator_;
  color_ = cv::Point3f((double)rand() / RAND_MAX, (double)rand() / RAND_MAX,
                       (double)rand() / RAND_MAX);
  pixelMap_ = pixelMap;
  lines_ = lines;
  first_ = cv::Point2f(0, 0);
  second_ = cv::Point2f(0, 0);
  members_.clear();
}

SCOPE::~LineSupport()
{
  // If the coin gets ever to a point where it is deleted it will happen here. Coins are reused when possible.
  if(hasCoin())
    delete coinptr_;

  members_.clear();
  (*lines_).remove(this);
}

void SCOPE::assignMember(const uint16_t& x, const uint16_t& y)
{

  // Don't do anything if belongs to the same support
  if ((*pixelMap_)(x, y)->lineSupport_ == this)
  {
    std::cout << "Assigned an already existing member\n";
    return;
  }

  // The support gets assigned to this pixel
  (*pixelMap_)(x, y)->lineSupport_ = this;

  // The pixel gets assigned to this support
  members_.push_back((*pixelMap_)(x, y));

  // Update image moments
  size_++;
  m00++;
  m10 += x;
  m01 += y;
  m11 += x * y;
  m20 += x * x;
  m02 += y * y;
}

void SCOPE::removeMember(const uint16_t& x, const uint16_t& y)
{

  // If there is nothing to remove, don't
  if ((*pixelMap_)(x, y)->lineSupport_ == nullptr)
  {
    std::cout << "Tried to remove a cluster with address of nullptr... This should not be possible;\n";
    return;
  }

  bool found = false;

  // Search members if any belongs to the searched position
  for (auto it = members_.begin(); it != members_.end(); ++it)
  {

    // If found, break
    if ((*it) == (*pixelMap_)(x, y))
    {
      members_.erase(it);

      found = true;
      break;
    }
  }

  if (found)
  {

    (*pixelMap_)(x, y)->lineSupport_ = nullptr;

    // Update image moments
    size_--;
    m00--;
    m10 -= x;
    m01 -= y;
    m11 -= x * y;
    m20 -= x * x;
    m02 -= y * y;

    if (size_ <= 0)
      delete this;
  }
  else
  {
    std::cout << "not found..."
              << "\n";
  }
}

std::list<LineLevelPixel*>& SCOPE::getMembers() { return members_; }

void SCOPE::calculateLine()
{

  if (size_ <= 0)
  {
    std::cout << "\033[1;31m Error: At this point a zero size should be "
                 "illegal...\n \033[0m\n ";
    throw std::exception();
  }

  /*
   * The below is gotten directly from the paper
   * 'Image Moments-Based Structuring and Tracking of Objects'
   * by Lourena Rocha and Luiz Velho
   */

  x_center_ = getCenterX();
  y_center_ = getCenterY();

  float_t a = ((float_t)m20 / (float_t)m00) - (x_center_ * x_center_);
  float_t b = 2.0f * (((float_t)m11 / (float_t)m00) - (x_center_ * y_center_));
  float_t c = ((float_t)m02 / (float_t)m00) - (y_center_ * y_center_);

  float_t orientation = 0.5f * atan2(b, a - c);
  // orientation_ = orientation;

  float_t length = sqrt(6.0f * (a + c + sqrt(b * b + (a - c) * (a - c))));
  float_t minor_axis = sqrt(6.0f * (a + c - sqrt(b * b + (a - c) * (a - c))));

  direction_ = cv::Point2f(cos(orientation), sin(orientation));

  float_t t_x = direction_.x * length * 0.5f;
  float_t t_y = direction_.y * length * 0.5f;

  first_ = cv::Point2f((x_center_ + t_x), (y_center_ + t_y));
  second_ = cv::Point2f((x_center_ - t_x), (y_center_ - t_y));

  // Calculate ratio of major axis to minor axis
  major_axis_ = length;
  minor_axis_ = minor_axis;//椭圆的长轴和短轴
}

void SCOPE::merge(LineSupport* ls)
{

  // Don't merge if belongs to the same support
  if (ls == this)
  {
    return;
  }

  if (ls == nullptr)
  {
    std::cout << "\033[1;31m Error: Tried to merge with a non-existing "
                 "support...\n \033[0m\n ";
    throw std::exception();
  }

  // Get ID of the bigger support region
  // This gets rid of flickering in color and preserves track (only graphical)
  if (ls->getSize() > this->size_)
  {    
    this->color_ = ls->getColor();
    this->id_ = ls->getId();
  }

  auto tmp = ls->getMembers();
  for (auto it = tmp.begin(); it != tmp.end(); ++it)
  {

    // The support gets assigned to this pixel
    (*pixelMap_)((*it)->x, (*it)->y)->lineSupport_ = this;

    // The pixel gets assigned to this support
    members_.push_back((*pixelMap_)((*it)->x, (*it)->y));
  }

  // Update image moments
  size_ += ls->getSize();
  m00 += ls->getM00();
  m10 += ls->getM10();
  m01 += ls->getM01();
  m11 += ls->getM11();
  m20 += ls->getM20();
  m02 += ls->getM02();

  // If ls contains a tracked coin, give coin to this cluster
  if(ls->hasCoin()){
    ls->handoverCoin(this);
  }

  delete ls;
}

cv::Point3f SCOPE::getColor() { return color_; }

uint64_t SCOPE::getId() { return id_; }

uint32_t SCOPE::getSize() {

  return size_; }

void SCOPE::recalculateAngle()
{

  float angle = 0;

  float x = 0;
  float y = 0;

  for (auto it = members_.begin(); it != members_.end(); ++it)
  {

    // Check exception
    if ((*pixelMap_)((*it)->x, (*it)->y)->angle_ == 5.0)
    {
      std::cout << "\033[1;31m Error: At least one member had an angle of type "
                   "INVALID_ANGLE which should be impossible...\n \033[0m\n ";
      throw std::exception();
    }

    if ((*pixelMap_)((*it)->x, (*it)->y)->angle_ > 1.0 ||
        (*pixelMap_)((*it)->x, (*it)->y)->angle_ < 0.0)
      std::cout << "Angle was invalid ... "
                << (*pixelMap_)((*it)->x, (*it)->y)->angle_
                << " at :" << (*it)->x << " " << (*it)->y << "\n";


    angle = (*pixelMap_)((*it)->x, (*it)->y)->angle_* PI;

    // We don't want wrong intermediates, therefore switch angles close to horizontal
    if((*pixelMap_)((*it)->x, (*it)->y)->angle_>= 1.0-MAX_ERROR){
      x -= cos(angle);
      y -= sin(angle);
    }
    else{

      x += cos(angle);
      y += sin(angle);
    }
  }

  // Calculate intermediate angle
  angle_ = atan2(y, x) * INV_PI;

  if (angle_ < 0.0)
  {
    angle_ = angle_ + 1.0;
  }
  if (angle_ > 1.0)
  {
    angle_ = angle_ - 1.0;
  }
}


float_t SCOPE::getDistanceTo(LineSupport* ls)
{

  // Get distance from cluster center to other cluster
  // center projected onto first clusters normal axis

  float_t dx = this->getCenterX() - ls->getCenterX();
  float_t dy = this->getCenterY() - ls->getCenterY();

  return fabs(-dx * sin(angle_ * PI) + dy * cos(angle_ * PI));
}

float_t SCOPE::getCenterX() { return (float_t)m10 / (float_t)m00; }

float_t SCOPE::getCenterY() { return (float_t)m01 / (float_t)m00; }

float SCOPE::getAngle() { return angle_; }

uint64_t SCOPE::getM00() { return m00; }

uint64_t SCOPE::getM01() { return m01; }

uint64_t SCOPE::getM10() { return m10; }

uint64_t SCOPE::getM11() { return m11; }

uint64_t SCOPE::getM20() { return m20; }

uint64_t SCOPE::getM02() { return m02; }

cv::Point SCOPE::getFirst() { return first_; }

cv::Point SCOPE::getSecond() { return second_; }

void SCOPE::setAge(uint64_t age) { age_ = age; }

uint64_t SCOPE::getAge() { return age_; }

cv::Point2f SCOPE::getCentroid(){return cv::Point2f(x_center_, y_center_); }

cv::Point2f SCOPE::getDirection(){ return direction_; }

bool SCOPE::hasCoin(){  return coinptr_ == nullptr? false:true;}

void SCOPE::setCoin(Coin*& coin){

  coinptr_ = coin;
  coinptr_->setPlace(this);

}

void SCOPE::handoverCoin(LineSupport* ls){

 if(ls->hasCoin()){
   delete ls->coinptr_;
   (ls->coinptr_->owner_) = &coinptr_;
 }

  ls->setCoin(coinptr_);
  coinptr_ = nullptr;
}

Coin*& SCOPE::trackCoin(){

  return coinptr_;
}

#undef SCOPE
}
