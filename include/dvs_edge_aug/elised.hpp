/*
* ELiSeD++ is based on the original ELiSeD algorithm presented by Braendli/Scaramuzza.
*
* Copyright (C) 2016 Andrea Luca Lampart <lamparta at student dot ethz dot ch> (ETH Zurich)
* For more information see <https://github.com/andrealampart/elised_plus_plus>
*
* ELiSeD++ is free software: you can redistribute it and/or modify
* it under the terms of the GNU General Public License as published by
* the Free Software Foundation, either version 3 of the License, or
* (at your option) any later version.
*
* ELiSeD++ is distributed in the hope that it will be useful,
* but WITHOUT ANY WARRANTY; without even the implied warranty of
* MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
* GNU General Public License for more details.
*
* You should have received a copy of the GNU General Public License
* along with ELiSeD++. If not, see <http://www.gnu.org/licenses/>.
*/

#pragma once

#include <stdint.h>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <eigen3/Eigen/Core>
#include <mutex>

#include "dvs_edge_aug/eventbuffer.hpp"
#include "dvs_edge_aug/tic_toc.hpp"
#include "dvs_edge_aug/line_support.hpp"
#include <fstream>

#define EL_RED_BLUE       1
#define EL_GRAYSCALE      2
#define EL_MASK_SMALL     4
#define EL_MASK_LARGE     8
#define EL_VISUALIZE      16
#define EL_SOBEL          32
#define EL_SCHARR         64




namespace event_mapping{

template <uint16_t COLS, uint16_t ROWS>
class Elised{

public:

  Elised(int32_t config, float_t max_dist = 6.0, float_t max_err = 0.1, int8_t min_cand = 1, int8_t min_display_size = 15 );
  ~Elised();


  /*!
   * \brief Push new events (Can be implemented as callback via std::function...)
   * @param x         x coordinate
   * @param y         y coordinate
   * @param polarity  event polarity
   * @param timestamp event timestamp
   */
  void push(const uint16_t &x, const uint16_t &y, const bool &polarity, const uint64_t &timestamp);

  /*!
   * \brief Render preview images
   */
  void render();



  /*!
   * \brief Returns the integrated image (needs to be rendered first)
   * \return
   */
  cv::Mat getVisualizedIntegrated();

  /*!
   * \brief Returns Elised visualization (needs to be rendered first)
   * \return
   */
  cv::Mat getVisualizedElised();

  /*!
   * \brief Accesses the Line Support at pixel x, y and returns its centroid
   * \param x
   * \param y
   * \param centroid
   */
  bool getCentroidAt(const uint16_t &x, const uint16_t &y, cv::Point2f& centroid);

  /*!
   * \brief Accesses the Line Support at pixel x, y and returns it's direction
   * \param x
   * \param y
   * \param direction
   */
  bool getDirectionAt(const uint16_t &x, const uint16_t &y, cv::Point2f& direction);

  /*!
   * \brief Accesses the Line Support at pixel x, y and returns its pointer
   * \param x
   * \param y
   * \param LineSupport*
   */
  bool getPointerAt(const uint16_t &x, const uint16_t &y, LineSupport* &ls);

  /*!
   * \brief Uses the precalculated undistortion map to remap points
   * @param x_dist  distorted x coordinate
   * @param y_dist  distorted y coordinate
   */
  cv::Point inline undistort(const uint16_t &x_dist, const uint16_t &y_dist);

  /*!
   * \brief Returns all lines bigger than a certain threshold inside the scene
   * \return
   */
  Eigen::MatrixXd getLines();

  bool placeCoin(const uint16_t &x, const uint16_t &y, Coin* &coin);


protected:

  /*!
   * lineSupports
   */
  cv::Mat lineSupportVis_;

  /*!
   * Event image (should not be visualized)
   */
  cv::Mat integratedFrame_;

  /*!
   * \brief integrated image used for visualization of events at frame rate
   */
  cv::Mat integrated_;



  /*!
   * \brief Applies Gradient Detection on each event using timestamp image
   * @param x     x coordinate
   * @param y     y coordinate
   */
  void inline perPointGradients(const uint16_t &x, const uint16_t &y);

  /*!
   * \brief Removes oldest events in buffer
   * @param event  oldest buffered event
   */
  void inline remove(const uint16_t &x, const uint16_t &y);

  /*!
   * \brief Creates a mapping from distorted to undistorted points
   */
  void inline createUndistortionMap();

  /*!
   * \brief assignLineSupport
   * \param x
   * \param y
   */
  void inline assignLineSupport(const uint16_t &x, const uint16_t &y);

  /*!
   * \brief checks if angles of two pixels differ by more than a defined value
   * \param x
   * \param y
   * \param x_target
   * \param y_target
   */
  bool inline checkAngle(const uint16_t &x, const uint16_t &y, const uint16_t &x_target, const uint16_t &y_target);




   /*!
    * C array of Eigen 2D Array holding the timestamps of different polarity images
    * This may not be an elegant solution but works anyway. Might need to change later.
    */
   Eigen::Array<int64_t, ROWS, COLS> latestTimestamps_;

   Eigen::Array<int64_t, ROWS, COLS> gradX_;
   Eigen::Array<int64_t, ROWS, COLS> gradY_;


   Eigen::Array<int64_t, ROWS,COLS> gradients_;

   /*!
    * Map storing all angles
    */
   Eigen::Array<float, ROWS, COLS> angles_;

   /*!
    * Lifetime gradient image Visualization
    */
   cv::Mat lifetimeGradients_;

   /*!
    * Lifetime Angle image Visualization
    */
   cv::Mat lifetimeAngles_;

   /*!
    * Harris Corners
    */
   cv::Mat harrisCorners_;



   std::list<LineSupport*> lines_;

   /*!
    * Buffer storing the latest n events
    */
   EventBuffer<cv::Point2i> buffer_;

   /*!
    * Sobel Masks
    */
   Eigen::Array<int64_t, 3, 3> filterMaskXSmall_;
   Eigen::Array<int64_t, 3, 3> filterMaskYSmall_;

   Eigen::Array<int64_t, 5, 5> filterMaskX_;
   Eigen::Array<int64_t, 5, 5> filterMaskY_;

  //  /*
  //   * Gaussian filter
  //   */
  //  float gaussFilter_[];

  Eigen::Array< LineLevelPixel*,ROWS,COLS> pixelMap_;

  //Matrices

  cv::Mat cameraMatrix_;
  cv::Mat distCoeffs_;
  cv::Mat undistortionMap_;

  enum DisplayType {RED_BLUE,GRAYSCALE} ;
  enum KernelSize {SMALL, LARGE};
  enum KernelType {SOBEL,SCHARR} ;

  // Parameters
  struct PARAMETER_TYPE{
    float minAngleDiff_;
    float maxDistance_;
    float maxError_;            // Maximum error angle to count as candidate
    int minCandidates_;         // Minimum candidates required for a cluster to grow
    uint16_t minDisplaySize_;   // Minimum size for a cluster to be displayed
    uint8_t display_type_;      // display type: red/blue or greyscale
    uint8_t kernel_size_;
    uint8_t kernel_type_;
    bool visualizeIntegrated_;
    std::string write_dir_;
  }params;

  std::mutex eventMutex_;

  /*
  * Gaussian filter
  */
  float gaussFilter_[];
};

}
#include "dvs_edge_aug/elised.inl"
