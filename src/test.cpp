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
    is >> new_event.timestamp >> new_event.x >> new_event.y >> new_event.polarity;
    return is;
}

/*
 *
 *  main()
 * 
 */
int main(int argc, char const *argv[])
{
    ifstream fin("~/Data/shapes_6dof/events.txt");
    Event new_event;
    int counter = 0;
    cout<<"Start reading file\n";
    while(read_event(fin, new_event))
    {
        counter++;
        cout<<counter<<" "<<new_event.timestamp<<" "<<new_event.x<<endl;
        if(counter>1000)
            break;        
        Elised<240,180> line_detector(0xff);
        line_detector.push(new_event.x, new_event.y, new_event.polarity, new_event.timestamp);


    }
    return 0;
}


