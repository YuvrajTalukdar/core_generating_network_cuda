#include<iostream>
#include<string.h>
#include<dirent.h>
#include<vector>
#include<thread>
#include<iomanip>//for erasing the above line
#include<sys/ioctl.h>//for the terminal size

//#include"core_start.h"

using namespace std;

void clrscr()
{
    cout << "\033[2J\033[1;1H";
}

string select_network_file()
{
    struct dirent *de;
    DIR *dr = opendir("./trained_network/");
    if (dr == NULL)
    {
        cout<<"Could not open current directory";
        return "";
    }
    vector<string> network_save_file;
    network_save_file.clear();
    while ((de = readdir(dr)) != NULL)
    {
        //cout<<"\nname="<<de->d_name;
        if(strcasestr(de->d_name,"segment_")!=NULL)//strcasestr returns only char*
        {   network_save_file.push_back(de->d_name);}
    }
    closedir(dr);
    
    cout<<"\n\nChoose the segment save folder:-\n";
    for(int a=0;a<network_save_file.size();a++)
    {   cout<<a+1<<". "<<network_save_file[a]<<endl;}
    if(network_save_file.size()==0)
    {   cout<<"\nERROR No segment folder found in the current directory!!!\n";}
    cout<<"\npress 0 to go back\n";
    point1:
    cout<<"\nEnter your option: ";
    int option2;
    cin>>option2;
    if(option2>network_save_file.size() || option2<0)
    {
        cout<<"Wrong option!!!";
        goto point1;
    }
    else if(option2==0)
    {   return "";}
    else
    {   return "./trained_network/"+network_save_file[option2-1];}
}

string select_data_file(vector<string>& csv_save_file)
{
    cout<<"\n\nList of available csv files for input data in the current directory:-\n";
    for(int a=0;a<csv_save_file.size();a++)
    {   cout<<a+1<<". "<<csv_save_file[a]<<endl;}   
    if(csv_save_file.size()==0)
    {   cout<<"\nERROR No .csv file found in the current directory!!!\n";}
    cout<<"\nPress 0 to go back"; 
    int file_choice;
    point2:
    cout<<"\n\nSelect your option: ";
    cin>>file_choice;
    if(file_choice>csv_save_file.size() || file_choice<0)
    {   
        cout<<"wrong option!!!";
        goto point2;
    }
    else if(file_choice==0)
    {   
        clrscr();
        return "";
    }
    else
    {   return csv_save_file[file_choice-1];}
}

int get_batch_size()
{
    int batch_size;
    cout<<"\nEnter Batch Size (This is for preventing the algorithm from using too much RAM/VRAM. Try setting the value to 100): ";
    cin>>batch_size;
    return batch_size;
}

bool menu(string &file_name,int &test_train_predict,string &network_save_file_name,int &batch_size)
{
    //.csv file finder
    struct dirent *de;  // Pointer for directory entry
    // opendir() returns a pointer of DIR type. 
    DIR *dr = opendir("./data/");
    if (dr == NULL)  // opendir returns NULL if couldn't open directory
    {
        cout<<"Could not open current directory";
        return false;
    }
    char filename_sub_str[]=".csv";
    vector<string> csv_save_file;
    csv_save_file.clear();
    while ((de = readdir(dr)) != NULL)
    {
        if(strcasestr(de->d_name,filename_sub_str) && strcasestr(de->d_name,"network-")==NULL && strcasestr(de->d_name,"core-")==NULL)//strcasestr returns only char*
        {   csv_save_file.push_back(de->d_name);}
    }
    closedir(dr);
    //home page
    struct winsize w;
    ioctl(0,TIOCGWINSZ,&w);
    point0:
    clrscr();
    string heading="CORE GENERATING NETWORK TEST INTERFACE\n";
    cout<<setw(w.ws_col/2+heading.length()/2)<<heading;
    cout<<"\n\n1. Train and test the network\n"<<"2. Use a trained network to make predictions\n";
    cout<<"3. Enter individual data and predict the results using a trained network\n4. Test an already trained network.\n";
    cout<<"\nPress 0 to close\n\n";
    point1:
    cout<<"Select your option: ";
    int option1;
    cin>>option1;
    if(option1>4 || option1<0)
    {
        cout<<"Wrong option!!!\n";
        goto point1;
    }
    else if(option1==1)
    {
        //file selector for part training and part testing
        clrscr();
        heading="CORE GENERATING NETWORK TEST INTERFACE\n";
        cout<<setw(w.ws_col/2+heading.length()/2)<<heading;
        string note1="\n\nHere a new network is created and trained and tested on the dataset you selected. At max 1/2 of the data can be used for training and rest will be used for testing. The training data may be less than 1/2 of the entire dataset, and rest will be for testing.";
        cout<<"\n\nNOTE:-"<<note1;    
        file_name=select_data_file(csv_save_file);
        if(file_name.length()==0)
        {   
            clrscr();
            goto point0;
        }
        file_name="./data/"+file_name;
        clrscr();
        heading="CORE GENERATING NETWOR TEST INTERFACE\n";
        cout<<setw(w.ws_col/2+heading.length()/2)<<heading;
        test_train_predict=1;
    
        batch_size=get_batch_size();
    }
    else if(option1==2)
    {
        point4:
        clrscr();
        heading="CORE GENERATING NETWORK TEST INTERFACE\n";
        cout<<setw(w.ws_col/2+heading.length()/2)<<heading;  
        string note2="\n\nThe prediction results will be saved in a file named prediction_results.csv. ";
        cout<<"\n\nNOTE:-"<<note2; 
        network_save_file_name=select_network_file();
        if(network_save_file_name.length()==0)
        {   goto point0;}
        clrscr();
        heading="CORE GENERATING NETWORK TEST INTERFACE\n";
        cout<<setw(w.ws_col/2+heading.length()/2)<<heading;  
        file_name=select_data_file(csv_save_file);
        if(file_name.length()==0)
        {   goto point4;}
        file_name="./data/"+file_name;

        test_train_predict=2;
    }
    else if(option1==3)
    {
        clrscr();
        heading="CORE GENERATING NETWORK TEST INTERFACE\n";
        cout<<setw(w.ws_col/2+heading.length()/2)<<heading;  
        string note2="\n\nHere you can enter the data yourself and let the algorithm predict the appropriate result, \nthe result will appear instantaneously in this interface itself.";
        cout<<"\n\nNOTE:-"<<note2; 
        //.network file finder
        network_save_file_name=select_network_file();
        if(network_save_file_name.length()==0)
        {   goto point0;}
        
        test_train_predict=3;//a new option 
    }
    else if(option1==4)
    {
        point2:
        clrscr();
        heading="CORE GENERATING NETWORK TEST INTERFACE\n";
        cout<<setw(w.ws_col/2+heading.length()/2)<<heading;      
        string note="\n\nHere an existing network will be used to test the accuracy that can be achieved by it.";
        cout<<"\n\nNOTE:-"<<note;
        //.network file finder
        network_save_file_name=select_network_file();
        if(network_save_file_name.length()==0)
        {   goto point0;}
        clrscr();
        heading="CORE GENERATING NETWORK TEST INTERFACE\n";
        cout<<setw(w.ws_col/2+heading.length()/2)<<heading;  
        file_name=select_data_file(csv_save_file);
        if(file_name.length()==0)
        {   goto point2;}
        file_name="./data/"+file_name;

        test_train_predict=4;
    }
    else if(option1==0)
    {   test_train_predict=-1;}
    return false;
}