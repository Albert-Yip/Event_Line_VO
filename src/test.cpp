#include <fstream>
#include <sstream>
#include <iostream>
#include <stdint.h>
#include "dvs_edge_aug/elised.hpp"

using namespace std;
using namespace event_mapping;

struct Event
{
    uint16_t x;
    uint16_t y;
    bool polarity;
    uint64_t timestamp;
};

istream &read_event(istream &is, Event &new_event)
{
    double tmp_timestamp;
    is >> tmp_timestamp >> new_event.x >> new_event.y >> new_event.polarity;
    new_event.timestamp = uint64_t (tmp_timestamp*1e9);
    // new_event.x = new_event.x % 128;
    // new_event.y = new_event.y % 128;
    cout<<new_event.timestamp<<","<<new_event.x<<","<<new_event.y<<","<<new_event.polarity<<endl;
    return is;
}

/*
 *
 *  main()
 * 
 */
int main(int argc, char const *argv[])
{
    // ifstream fin("/home/albert/Data/shapes_6dof/events.txt");
    ifstream fin("/home/albert/Data/poster_6dof/events.txt");
    Event new_event;
    int counter = 0;
    cout<<"Start reading file\n";
    Elised<128,128> line_detector(0xff);
    while(read_event(fin, new_event))
    {
        counter++;
        // cout<<counter<<" "<<new_event.timestamp<<" "<<new_event.x<<endl;
        if(counter>100000)
            break;        
        // Elised<240,180> line_detector(0xff);
        if(new_event.x >= 128 || new_event.y >= 128)
            continue;

        line_detector.push(new_event.x, new_event.y, new_event.polarity, new_event.timestamp);
    }
    line_detector.render();
    // line_detector.getVisualizedIntegrated();
    // line_detector.getVisualizedElised();
    cv::namedWindow("win_1", CV_WINDOW_NORMAL); 
    cv::namedWindow("win_2", CV_WINDOW_NORMAL); 
    cv::imshow("win_1",line_detector.getVisualizedIntegrated());
    cv::imshow("win_2",line_detector.getVisualizedElised());
    cv::waitKey();
    return 0;
}


