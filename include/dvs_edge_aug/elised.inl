/*
* ELiSeD++ is based on the original ELiSeD algorithm presented by Braendli/Scaramuzza.
*
* Copyright (C) 2016 Andrea Luca Lampart <lamparta at student dot ethz dot ch> (ETH Zurich)
* For more information see <https://github.com/andrealampart/elised_plus_plus>
*/

#include <stdint.h>
#include <exception>
#include <eigen3/Eigen/Core>
#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/calib3d/calib3d.hpp>
#include "opencv2/highgui/highgui.hpp"
#include "dvs_edge_aug/line_support.hpp"
#include "dvs_edge_aug/tic_toc.hpp"



namespace event_mapping{

#define TMP template<uint16_t COLS, uint16_t ROWS>
#define SCOPE Elised<COLS,ROWS>
#define INVALID_ANGLE 5.0

  TMP
  SCOPE::Elised(int32_t config, float_t max_dist, float_t max_err, int8_t min_cand, int8_t min_display_size )
    :buffer_(EventBuffer<cv::Point2i>(EB_LARGE))
  {


    // Parameters
    if((config & EL_RED_BLUE) == EL_RED_BLUE){
      params.display_type_ = RED_BLUE;
    }
    else if((config & EL_GRAYSCALE) == EL_GRAYSCALE){
      params.display_type_ = GRAYSCALE;
    }
    else{
      params.display_type_ = RED_BLUE;
    }

    if((config & EL_MASK_SMALL) == EL_MASK_SMALL){
      params.kernel_size_ = SMALL;
    }
    else if((config & EL_MASK_LARGE) == EL_MASK_LARGE){
      params.kernel_size_ = LARGE;
    }
    else{
      params.kernel_size_ = SMALL;
    }


    if((config & EL_SOBEL) == EL_SOBEL){
      params.kernel_type_ = SOBEL;
    }
    else if((config & EL_SCHARR) == EL_SCHARR){
      params.kernel_type_ = SCHARR;
    }
    else{
      params.kernel_type_ = SCHARR;
    }

    params.maxDistance_ = max_dist;
    params.maxError_ = max_err;
    params.minCandidates_ = min_cand;
    params.minDisplaySize_ =  min_display_size;
    params.visualizeIntegrated_ = true;


    if(COLS < (params.kernel_size_? 5:3) || ROWS < (params.kernel_size_? 5:3)){

      std::cout<<"\033[1;31m Error: Elised ROWS and COLS are too small... initialize with at least 3x3 pixels \n \033[0m\n";
      throw std::exception();

    }



    // Create images
    if(params.display_type_ == RED_BLUE){

       integrated_ = cv::Mat(ROWS,COLS,CV_32FC3);
       integratedFrame_ = cv::Mat(ROWS,COLS,CV_32FC3);
       integratedFrame_ = cv::Scalar(0.5,0.5,0.5);

    }
    else{

       integrated_ = cv::Mat(ROWS,COLS, CV_8UC1);
       integratedFrame_ = cv::Mat(ROWS,COLS, CV_8UC1);
       integratedFrame_ = cv::Scalar(128);

    }


    // Images
    harrisCorners_ = cv::Mat(ROWS,COLS,CV_32FC3);
    harrisCorners_ = cv::Scalar(0.0,0.0,0.0);
    lineSupportVis_ = cv::Mat(ROWS, COLS, CV_32FC3);
    lineSupportVis_ = cv::Scalar(0.0,0.0,0.0);

    // Lifetimes Map
    latestTimestamps_ = Eigen::Array<int64_t,ROWS,COLS>::Constant(ROWS,COLS,0);



    // pixelmap
    for(uint16_t i = 0; i<COLS;++i){
      for(uint16_t j = 0; j<ROWS;++j){

        LineLevelPixel* ptr = new LineLevelPixel(nullptr,INVALID_ANGLE,i,j);
        pixelMap_(i,j) = ptr;
      }
    }

    // intrinsics dvs
    cameraMatrix_ = (cv::Mat_<double>(3,3) << 127.9115321697418, 0, 71.16039994925386, 0, 128.1735318288403, 74.01369983664087, 0, 0, 1);
    distCoeffs_ = (cv::Mat_<double>(1,5) << -0.3541172895426445, 0.1401730900933051, -0.0004481769257326341, 0.000241525353746612, 0);

    // Create undistortion map
    createUndistortionMap();

    // Create Derivative Filter Masks
    if(params.kernel_type_ == SOBEL){
      filterMaskXSmall_ << 1, 0,-1, 2, 0,-2, 1, 0,-1;
      filterMaskYSmall_ << 1, 2, 1, 0, 0, 0, -1, -2, -1;
      filterMaskX_ << 1, 2, 0, -2, -1, 4, 8, 0, -8, -4, 6, 12, 0, -12, -6, 4, 8, 0, -8, -4, 1, 2, 0, -2, -1;
      filterMaskY_ << 1, 4, 6, 4, 1, 2, 8, 12, 8, 2, 0, 0, 0, 0, 0, -2, -8, -12, -8, -2, -1, -4, -6, -4, -1;
    }
    else if(params.kernel_type_ == SCHARR){
      filterMaskXSmall_ << 1,0,-1,3,0,-3,1,0,-1;
      filterMaskYSmall_ << -1,-3,-1,0,0,0,1,3,1;
      filterMaskX_ << 1,1,0,-1,-1,2,2,0,-2,-2,3,6,0,-6,-3,2,2,0,-2,-2,1,1,0,-1,-1;
      filterMaskY_ << -1,-2,-3,-2,-1,-1,-2,-6,-2,-1,0,0,0,0,0,1,2,6,2,1,1,2,3,2,1;
    }



  }

  TMP
  SCOPE::~Elised(){



  }


  TMP
  void SCOPE::push(const uint16_t &x, const uint16_t &y, const bool &polarity, const uint64_t &timestamp){


  //Boundary check (for Sobel filtering)
  if(!((x > params.kernel_size_) && (x < (COLS-params.kernel_size_-1)) && (y > params.kernel_size_) && (y < (ROWS-params.kernel_size_-1))))
     return;



    // Remove oldest buffer element
    cv::Point2i oldest = buffer_.getOldest();
    remove(oldest.x, oldest.y);

    // Write event into buffer
    cv::Point2i e = {x,y};
    buffer_.write(e);

    // Save timestamps
    // CAUTION: In eigen, x and y are by default stored in column-major in order
    latestTimestamps_(x,y) = (int64_t)timestamp;


    if(params.visualizeIntegrated_){

       std::lock_guard<std::mutex> lock(eventMutex_);

    // Push to visualization image
      if(params.display_type_ == RED_BLUE){

        //cv::Point2i uP = undistort(x,y);

        // Push event into colored preview image
        integratedFrame_.at<cv::Point3f>(y,x) = (polarity == true ? cv::Point3f(1.0, 0, 0) : cv::Point3f(0,0,1.0));

      }
      else{
        // Push event into grayscale preview image
         integratedFrame_.at<uint8_t>(y,x) = (polarity == true ? 0 : 255);

      }
    }

    // Calculate gradients and angles from event
    perPointGradients(x,y);

    {
       std::lock_guard<std::mutex> lock(eventMutex_);

        // Assign a Line Support if available
        assignLineSupport(x,y);
    }


  }


  TMP
  void inline SCOPE::perPointGradients(const uint16_t &x, const uint16_t &y){

    std::lock_guard<std::mutex> lock(eventMutex_);

    // Temporary variables
    int64_t sum0X,sum0Y;



    //-------------------------------SOBEL FILTERING--------------------------------------------------

    // If sobel size is large, use a 5x5 window , else 3x3
    if(params.kernel_size_ == LARGE){

      Eigen::Array<int64_t,5,5> tmpBlock;


    // Do eigen coefficient-wise product for use of SSE instructions on matrix elements
     tmpBlock = (latestTimestamps_).template block<5,5>(x-2,y-2);
     sum0X = (tmpBlock*filterMaskX_).sum();

     tmpBlock = (latestTimestamps_).template block<5,5>(x-2,y-2);
     sum0Y = (tmpBlock*filterMaskY_).sum();


    }
    else{

        Eigen::Array<int64_t,3,3> tmpBlock;


      // Do eigen coefficient-wise product for use of SSE instructions on matrix elements
       tmpBlock = (latestTimestamps_).template block<3,3>(x-1,y-1);
       sum0X = (tmpBlock*filterMaskXSmall_).sum();

       tmpBlock = (latestTimestamps_).template block<3,3>(x-1,y-1);
       sum0Y = (tmpBlock*filterMaskYSmall_).sum();

    }

    // Angle is mapped from [0,pi) to [0,1)
    if(sum0Y<0){
      sum0Y = - sum0Y;
      sum0X = - sum0X;
    }

    // Set angle onto pixel
    pixelMap_(x,y)->angle_ = (float) atan2l(sum0Y,sum0X)*0.31830988618;



  }


  TMP
  void SCOPE::render(){



  if(params.visualizeIntegrated_){

      std::lock_guard<std::mutex> lock(eventMutex_);

   integratedFrame_.copyTo(integrated_);

    if(params.display_type_ == RED_BLUE ){
        integratedFrame_ = cv::Scalar(0.5,0.5,0.5);
     }
     else{
        integratedFrame_ = cv::Scalar(128);
     }
  }

   lineSupportVis_ = cv::Scalar(0.0,0.0,0.0);

  // Render Line Supports
  // CAUTION! Don't render if not needed
  /*
   * DVS events are faster processed than read for visualization.
   * This will cause the application to crash at some point due to overwrite
   */

   {
       std::lock_guard<std::mutex> lock(eventMutex_);

   for(int i =  0; i<COLS;++i){
      for(int j = 0; j<ROWS;++j){
        if((pixelMap_(i,j)->lineSupport_!= 0x0)){
          if(pixelMap_(i,j)->lineSupport_->getSize() > params.minDisplaySize_){

            lineSupportVis_.at<cv::Point3f>(j,i) = pixelMap_(i,j)->lineSupport_->getColor();
          }

        }
      }
    }

   for(auto it = lines_.begin(); it != lines_.end();++it){
      if((*it)!=nullptr){
        if((*it)->getSize()>params.minDisplaySize_ && (*it)->major_axis_ > 3* (*it)->minor_axis_){

          cv::Point first = (*it)->getFirst();
          cv::Point second = (*it)->getSecond();
          cv::line(lineSupportVis_,first,second,cv::Scalar(((*it)->getColor()).x,((*it)->getColor()).y,((*it)->getColor()).z),1);
        }
      }
     }
  }//lock_guard

  }

  TMP
  void inline SCOPE::remove(const uint16_t &x, const uint16_t &y){


    // Remove event from support or delete it's angle
    if(pixelMap_(x,y)->lineSupport_!=nullptr){

      pixelMap_(x,y)->lineSupport_->setAge(latestTimestamps_(x,y));
      pixelMap_(x,y)->lineSupport_->removeMember(x,y);

      pixelMap_(x, y)->angle_ = INVALID_ANGLE;
    }
    else{
      pixelMap_(x, y)->angle_ = INVALID_ANGLE;
    }

  }

  TMP
  cv::Point inline SCOPE::undistort(const uint16_t &x_dist, const uint16_t &y_dist){

    cv:: Point undistorted = undistortionMap_.at<cv::Point2d>(y_dist,x_dist);

    return undistorted;
  }

  TMP
  void inline SCOPE::createUndistortionMap(){

   undistortionMap_ = cv::Mat(ROWS,COLS,CV_64FC2);
    cv::Mat points_dist;
    cv::Mat points_undist;
    cv::Point2d point;


    // Loop over all pixel coordinates
    for(int i= 0;i<ROWS;++i){
      for(int j = 0;j<COLS;++j){

        // Remember, that OPENCV2 stores in row-major order
        point = cv::Point2f((double)j,(double)i);
        points_dist = cv::Mat(1,1,CV_64FC2, &point);

        // Undistort point
        cv::undistortPoints(points_dist,points_undist, cameraMatrix_, distCoeffs_);

        //Check if point will map onto outside of camera and mask by placing invalid value
        // (This might not be an elegant solution but sufficient for now)
        undistortionMap_.at<cv::Point2d>(i,j) = cv::Point2d(points_undist.at<cv::Point2d>(0,0).x*cameraMatrix_.at<double>(0,0)+cameraMatrix_.at<double>(0,2), points_undist.at<cv::Point2d>(0,0).y*cameraMatrix_.at<double>(1,1)+cameraMatrix_.at<double>(1,2));
       // if(undistortionMap_.at<cv::Point2d>(i,j).x < 0 || undistortionMap_.at<cv::Point2d>(i,j).x > (COLS-1) || undistortionMap_.at<cv::Point2d>(i,j).y < 0 || undistortionMap_.at<cv::Point2d>(i,j).y > (ROWS-1)){
         // undistortionMap_.at<cv::Point2d>(i,j) = cv::Point2d(-1,-1);
        //}


      }//for(j)

    }//for(i)



  }

  TMP
  void inline SCOPE::assignLineSupport(const uint16_t &x, const uint16_t &y){

   //Set candidates if their angle is closer than 22.5 degrees:

    std::vector<cv::Point2i> candidates;
    int nrCandidates = 0;

    // Point (x-1,y-1)
    if(checkAngle(x,y, (x-1), (y-1))){
      candidates.push_back(cv::Point2i(x-1,y-1));
      nrCandidates++;
    }

    // Point (x,y-1)
    if(checkAngle(x,y, (x), (y-1))){
      candidates.push_back(cv::Point2i(x,y-1));
      nrCandidates++;
    }

    // Point (x+1,y-1)
    if(checkAngle(x,y, (x+1), (y-1))){
      candidates.push_back(cv::Point2i(x+1,y-1));
      nrCandidates++;
    }

    // Point (x-1,y)
    if(checkAngle(x,y, (x-1), (y))){
      candidates.push_back(cv::Point2i(x-1,y));
      nrCandidates++;
    }

    // Point (x+1,y)
    if(checkAngle(x,y, (x+1), (y))){
      candidates.push_back(cv::Point2i(x+1,y));
      nrCandidates++;
    }

    // Point (x-1,y+1)
    if(checkAngle(x,y, (x-1), (y+1))){
      candidates.push_back(cv::Point2i(x-1,y+1));
      nrCandidates++;
    }

    // Point (x,y+1)
    if(checkAngle(x,y, (x), (y+1))){
      candidates.push_back(cv::Point2i(x,y+1));
      nrCandidates++;
    }

    // Point (x+1,y+1)
    if(checkAngle(x,y, (x+1), (y+1))){
      candidates.push_back(cv::Point2i(x+1,y+1));
      nrCandidates++;
    }


//std::cout<<"Candidates: "<<nrCandidates<<"\n";

//----------------------------------CANDIDATE SELECTION------------------------------------------

  //If at least n candidates are present
  if(nrCandidates > params.minCandidates_){

        // check for oldest line support among neighbours()
        LineSupport* oldest = nullptr;
        int64_t age = latestTimestamps_(x,y);

        for(auto it = candidates.begin(); it != candidates.end(); ++it){

          // If the candidate has a line support
          if(pixelMap_(it->x,it->y)->lineSupport_!=nullptr ){

            // If his age is smaller than the main pixels
           // if( latestTimestamps_(it->x, it->y) < age){
            if(pixelMap_(it->x,it->y)->lineSupport_->getAge() < age){

              // Set it as oldest support
              oldest = pixelMap_(it->x,it->y)->lineSupport_;

              //Update age
              age = latestTimestamps_(it->x,it->y);
            }
          }

        }

        //At this stage we will definitely have an oldest line support if there is one
        // The center pixel is not used for oldest support as lines have to move to be detected

//---------------------------------ASSIGNMENT TO OLDEST-----------------------------------------

        // If oldest already has line support-> add candidates to that line support
       if(oldest!=nullptr){

          //Assign line supports of all members
          for(auto it = candidates.begin(); it != candidates.end(); ++it){



            // If the member already has a support region
            if(pixelMap_(it->x, it->y)->lineSupport_!=nullptr){

              // Merge only if lines are collinear
              if(oldest->getDistanceTo(pixelMap_(it->x,it->y)->lineSupport_)<params.maxDistance_){

              // Merge this points members to the found support region's members
              oldest->merge(pixelMap_(it->x, it->y)->lineSupport_);

             }

            }

            // If it has no support region assigned, just add this member
            else{
              oldest->assignMember(it->x,it->y);
            }
          }

          // After all candidates were assigned, assign center
          if(pixelMap_(x,y)->lineSupport_!=nullptr){

              pixelMap_(x,y)->lineSupport_->removeMember(x,y);

          }

           oldest->assignMember(x,y);
           oldest->recalculateAngle();
           oldest->calculateLine();
        }

//------------------------------CREATE NEW LINE SUPPORT-------------------------------------------

        // Else create new line support
        else{

            // If close but patch has no line support yet, create new Line Support
            LineSupport* lineTmp = new LineSupport(&pixelMap_, &lines_);

            // Set age of cluster
            lineTmp->setAge(latestTimestamps_(x, y));

            //insert line into line list
            lines_.push_back(lineTmp);


            // Add all candidates to that support
            for(auto it = candidates.begin();it!= candidates.end();++it){
              if(pixelMap_(it->x,it->y)->lineSupport_!=nullptr){

                pixelMap_(it->x,it->y)->lineSupport_->removeMember(it->x,it->y);

              }

              lineTmp->assignMember(it->x,it->y);
            }

            // Add center pixel too and remove from its previous support if it has one
            if(pixelMap_(x,y)->lineSupport_!=nullptr){
              pixelMap_(x,y)->lineSupport_->removeMember(x,y);
            }

            lineTmp->assignMember(x,y);
            lineTmp->recalculateAngle();
            lineTmp->calculateLine();
        }
    }


  }


  TMP
  bool inline SCOPE::checkAngle(const uint16_t &x, const uint16_t &y, const uint16_t &x_target, const uint16_t &y_target){

   // Boundary check: This check is necessary to avoid including padded pixels
   if(!((x_target>params.kernel_size_+1) && (x_target<COLS-params.kernel_size_-2) && (y_target > params.kernel_size_+1) && (y_target < ROWS-params.kernel_size_-2)))
      return false;

   float diff;


    // If pixel belongs to a line support , use line support angle
    if(pixelMap_(x_target,y_target)->lineSupport_!= nullptr )
    {

      if(pixelMap_(x_target,y_target)->lineSupport_->getAngle() == INVALID_ANGLE){
       std::cout<<"\033[1;31m Error: Line Support without angle should not be possible\n \033[0m\n ";
       throw std::exception();
      }



      diff = fabs(pixelMap_(x_target,y_target)->lineSupport_->getAngle() - pixelMap_(x,y)->angle_);
    }
    // Else use pixels angle
    else{

      if(pixelMap_(x_target,y_target)->angle_ == INVALID_ANGLE){
        return false;
      }

      diff = fabs(pixelMap_(x_target, y_target)->angle_ - pixelMap_(x,y)->angle_);
    }

      if(diff > 0.5){
        if((1.0-diff) < params.maxError_)
          return true;
        else
          return false;
      }
      else{
        if(diff < params.maxError_)
          return true;
        else
          return false;
      }

  }



  TMP
  cv::Mat SCOPE::getVisualizedIntegrated(){
    std::lock_guard<std::mutex> lock(eventMutex_);
    return integrated_;
  }

  TMP
  cv::Mat SCOPE::getVisualizedElised(){
    std::lock_guard<std::mutex> lock(eventMutex_);
    return lineSupportVis_;
  }

  TMP
  bool SCOPE::getCentroidAt(const uint16_t &x, const uint16_t &y, cv::Point2f& centroid){
    std::lock_guard<std::mutex> lock(eventMutex_);

    if(pixelMap_(x,y)->lineSupport_!=nullptr){
      centroid = pixelMap_(x,y)->lineSupport_->getCentroid();
      return true;
    }
    else{
      return false;
    }




  }

  TMP
  bool SCOPE::getDirectionAt(const uint16_t &x, const uint16_t &y, cv::Point2f& direction){

    std::lock_guard<std::mutex> lock(eventMutex_);

    if(pixelMap_(x,y)->lineSupport_!=nullptr){
      direction = pixelMap_(x,y)->lineSupport_->getDirection();
      return true;
    }
    else{
      return false;
    }

  }

  TMP
  bool SCOPE::getPointerAt(const uint16_t &x, const uint16_t &y, LineSupport* &ls){

    std::lock_guard<std::mutex> lock(eventMutex_);

    if(pixelMap_(x,y)->lineSupport_!=nullptr){
      ls = pixelMap_(x,y)->lineSupport_;
      return true;
    }
    else{
      return false;
    }

  }

  TMP
  bool SCOPE::placeCoin(const uint16_t &x, const uint16_t &y, Coin* &coin){

    std::lock_guard<std::mutex> lock(eventMutex_);

    if(pixelMap_(x,y)->lineSupport_!=nullptr){
      if(pixelMap_(x,y)->lineSupport_->hasCoin()){

        coin = pixelMap_(x,y)->lineSupport_->trackCoin();

      }
      else{

        coin = new Coin();
        pixelMap_(x,y)->lineSupport_->setCoin(coin);

      }
      return true;
    }
    else{
      return false;
    }

  }


  TMP
  Eigen::MatrixXd SCOPE::getLines(){

    Eigen::Matrix<double, 3, Eigen::Dynamic > line_mat;

    // Vector to store points temporarily
    std::vector<cv::Point> vec;

    int counter = 0;

    // First loop is only read-out, as we don't want to lock our mutex for too long
    {

    std::lock_guard<std::mutex> lock(eventMutex_);



    for(auto it = lines_.begin(); it != lines_.end();++it){
       if((*it)!=nullptr){
         if((*it)->getSize()>params.minDisplaySize_ && (*it)->major_axis_ > 3* (*it)->minor_axis_){

           cv::Point first = (*it)->getFirst();
           cv::Point second = (*it)->getSecond();
           counter++;
           counter++;
         }
       }
     }
    }

    // Resize output matrix
    line_mat.setOnes(3,counter);

    // Second loop is undistortion of the points and placing into eigen array
    for(int i = 0; i< vec.size();++i){

      cv::Point point = undistort(vec[i].x, vec[i].y);

      // NOTE: In this case, block operations might even be worse in efficiency
      line_mat(0,i) = point.x;
      line_mat(1,i) = point.y;
    }

    return line_mat;

  }

#undef TMP
#undef SCOPE


}
