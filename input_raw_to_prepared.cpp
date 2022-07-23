/*
file modified of compability with shuttle_converted.csv
*/

#include<iostream>
#include<fstream>
#include<vector>
#include<string>
#include<string.h>
#include<algorithm>

//#include"core_class.h"
#include"segment_class.h"

using namespace std;

unsigned int iterations,population_size,mutation_percentage;

struct raw_data{
    vector<vector<float>> rawData;
};

bool check_for_missing_data(string line) //to remove the missing data.
{
    bool status=false;
    for(int a=0;a<line.size();a++)
    {
        if(line.at(a)=='?')
        {
            status=true;
        }
    }
    return status;
}

void breaker(raw_data* rw_data,string line)
{
    char num_char[20]={'\0'},ch[2];
    vector<float> one_row_of_data;
    if(check_for_missing_data(line)==false)
    {
        for(int a=0;a<line.size();a++)
        {
            if(line.at(a)==',')
            {
                float val = atof(num_char);//*100  
                //cout<<val<<endl;
                one_row_of_data.push_back(val);
                for(int b=0;b<20;b++){
                    num_char[b]='\0';
                }
                continue;
            }
            ch[0]=line.at(a);
            ch[1]='\0';
            strcat(num_char,ch);
        }
        one_row_of_data.push_back(atof(num_char));//*100
        rw_data->rawData.push_back(one_row_of_data);
    }
}

void read_data_from_file(raw_data* rw_data,string file_name)
{
    ifstream  raw_data_stream(file_name,ios::in);
    string line;
    //int count=0;
    while(raw_data_stream)
    {
        raw_data_stream>>line;
        if(raw_data_stream.eof())
        {   break;}
        //to remove the id line. //deleted if(count>0)
        breaker(rw_data,line);
        //count++; //for just counting the no of lines in the file //required if if(count>0) line to remove the id line
    }
}

void display_rw_data(raw_data* rw_data)
{
    for(int a=0;a<rw_data->rawData.size();a++)
    {
        for(int b=0;b<rw_data->rawData[a].size();b++)
        {
            cout<<rw_data->rawData[a][b]<<",";
        }
        cout<<endl;
    }
}

void data_filter(raw_data* filtered_data,string file_name)
{
    raw_data rw_data;
    read_data_from_file(&rw_data,file_name);
    //display_rw_data(&rw_data);
    vector<float> row;
    for(int a=0;a<rw_data.rawData.size();a++)
    {
        row.clear();
        for(int b=0;b<rw_data.rawData[a].size();b++)
        {
            //b>0 option removes the id column //deleted if(b>0)
            row.push_back(rw_data.rawData[a][b]);
        }
        filtered_data->rawData.push_back(row);
    }
}

void prepare_data(nn_core_data_package_class* data_pack,string file_name,int test_train_predict)
{
    raw_data filtered_data;
    data_filter(&filtered_data,file_name);
    //display_rw_data(&filtered_data);

    vector<float> row;
    //this part pushes the data in the data vector(vector<vector<float>>) and lebel(vector<float>) in the label vector
    for(int a=0;a<filtered_data.rawData.size();a++)
    {
        row.clear();
        for(int b=0;b<filtered_data.rawData[a].size();b++)
        {
            if(test_train_predict==2)
            {   row.push_back(filtered_data.rawData[a][b]);}
            else
            {
                if(b==filtered_data.rawData[a].size()-1)
                {   data_pack->labels.push_back(filtered_data.rawData[a][b]);}
                else
                {   row.push_back(filtered_data.rawData[a][b]);}
            }
        }
        data_pack->data.push_back(row);
    }
}

void display_prepared_data(nn_core_data_package_class* data_pack,string file_name)
{cout<<"\n\nhello= "<<file_name<<endl;;
    for(int a=0;a<data_pack->data.size();a++)
    {
        //if(data_pack->labels[a]==8){ //for checking each of the labels
        for(int b=0;b<data_pack->data[a].size();b++)
        {
            cout<<data_pack->data[a][b]<<",";
        }
        cout<<" label= "<<data_pack->labels[a]<<endl;
        //}
    }
}

bool get_true_false(string msg)
{
    bool value;
    char option;
    point1:
    cout<<msg;
    cin>>option;
    if(option=='y'||option=='Y')
    {   value=true;}
    else if(option=='n'||option=='N')
    {   value=false;}
    else
    {   cout<<"\nWrong Option!!";goto point1;}
    return value;
}
chromosome get_critical_variables_from_user(unsigned int &iterations,unsigned int &population_size,unsigned int &mutation_percentage)
{
    chromosome critical_variables;
    char option;
    point1:
    cout<<"\nDo you want to enter the critical variables?(y/n) ";
    cin>>option;
    if(option=='y'||option=='Y')
    {   
        critical_variables.id=0;
        point2:
        critical_variables.flatening_fx_enabled=get_true_false("\nflatening_fx_enabled (y/n): ");
        critical_variables.zero_weight_remover=get_true_false("zero_weight_remover (y/n): ");
        critical_variables.extreame_weight_remover=get_true_false("extreame_weight_remover (y/n): ");
        cout<<"fp_change_value: ";
        cin>>critical_variables.fp_change_value;
        cout<<"summation_temp_threshold: ";
        cin>>critical_variables.summation_temp_thershold;
        cout<<"rhs_upper: ";
        cin>>critical_variables.rhs_upper;
        cout<<"rhs_lower: ";
        cin>>critical_variables.rhs_lower;
        cout<<"attributes_per_core: ";
        cin>>critical_variables.attributes_per_core;
        cout<<"data_division: ";
        cin>>critical_variables.data_division;
        point3:
        cout<<"\nDo you confirm the critical variables?(y/n) ";
        cin>>option;
        if(option=='y'||option=='Y')
        {}
        else if(option=='n'||option=='N')
        {   goto point2;}
        else
        {   cout<<"\nWrong Option!!";goto point3;}
    }
    else if(option=='n'||option=='N')
    {   critical_variables.id=-1;}
    else
    {   cout<<"\nWrong Option!!";goto point1;}
    if(critical_variables.id==-1)
    {
        cout<<"\nNo of iterations: ";
        cin>>iterations;
        cout<<"\nInitial population size: ";
        cin>>population_size;
        cout<<"\nmutation percentage: ";
        cin>>mutation_percentage;
    }

    return critical_variables;
}

datapack_structure_defination datapack_analyzer(nn_core_data_package_class* data_pack)
{
    datapack_structure_defination ds;    
    ds.no_of_elements_in_each_record=data_pack->data[0].size();
    vector<int> labels;
    bool found=false;
    for(int a=0;a<data_pack->data.size();a++)
    {
        found=false;
        for(int b=0;b<labels.size();b++)
        {
            if(labels[b]==data_pack->labels[a])
            {
                found=true;
                break;
            }
        }
        if(found==false)
        {
            labels.push_back(data_pack->labels[a]);
        }
    }
    ds.no_of_labels=labels.size();
    for(int a=0;a<labels.size();a++)
    {   ds.elements.push_back(labels[a]);}
    data_pack->analyze_status=true;

    return ds;
}

struct shuffling_data{
    vector<float> temp_data;
    int temp_label;
};
void shuffler(nn_core_filtered_data* f_data)
{
    vector<shuffling_data> shuffling_data_temp_vector;
    shuffling_data shuffling_data_temp;
    shuffling_data_temp_vector.clear();
    //pushing the data the the shuffling vector
    for(int a=0;a<f_data->data.size();a++)
    {
        shuffling_data_temp.temp_data=f_data->data[a];
        shuffling_data_temp.temp_label=f_data->label;
        shuffling_data_temp_vector.push_back(shuffling_data_temp);
    }
    //shuffeling the data
    random_shuffle(shuffling_data_temp_vector.begin(),shuffling_data_temp_vector.end());
    //pushing the data in the f_data
    f_data->data.clear();
    for(int a=0;a<shuffling_data_temp_vector.size();a++)
    {
        f_data->data.push_back(shuffling_data_temp_vector[a].temp_data);
        f_data->label=shuffling_data_temp_vector[a].temp_label;
    }
}

int filter(nn_core_data_package_class& data_pack,datapack_structure_defination ds,vector<nn_core_filtered_data>& f_data_vector)
{
    nn_core_filtered_data f_data;
    f_data_vector.clear();
    for(int a=0;a<ds.elements.size();a++)
    {
        f_data.data.clear();
        for(int b=0;b<data_pack.labels.size();b++)
        {
            if(ds.elements[a]==data_pack.labels[b])
            {
                f_data.data.push_back(data_pack.data[b]);
                f_data.label=data_pack.labels[b];
            }
        }
        f_data_vector.push_back(f_data);
    }
    //calling the shuffler
    for(int a=0;a<f_data_vector.size();a++)
    {
        shuffler(&f_data_vector[a]);//has same size as that is of no of element
    }
    int least_data=f_data_vector[0].data.size();
    for(int a=1;a<f_data_vector.size();a++)
    {
        if(least_data>f_data_vector[a].data.size())
        {   least_data=f_data_vector[a].data.size();}
    }
    if(least_data/6<6)
    {   return 2;}
    else
    {   return 6;}
}

void start_segment(
    int batch_size,
    int train_test_predict,
    nn_core_data_package_class& data_pack,
    string segment_dir,
    chromosome critical_variables)
{
    if(train_test_predict==2)//print predictions using already trained network
    {
        segment_class segment1(0,0,"default_segment");
        if(segment1.load_segment(segment_dir)==true)
        {   
            if(data_pack.data[0].size()==segment1.return_ds().no_of_elements_in_each_record)
            {   segment1.print_prediction(data_pack,train_test_predict);}
            else
            {   cout<<"\nERROR!!! Data Pack do not belongs to the loaded network.";}
        }
        else
        {   cout<<"\nERROR!!! failed to load network from the network file.";}
    }
    else if(train_test_predict==3)//making prediction on user entered individual data.
    {
        segment_class segment1(0,0,"default_segment");
        bool network_load_status=segment1.load_segment(segment_dir);
        if(network_load_status==true)
        {   
            cout<<"\nNetwork successfully loaded";
            segment1.make_prediction_on_user_entered_data();
        }
        else
        {   cout<<"\nERROR!!! failed to load network from the network file.";}
    }
    else if(train_test_predict==4 || train_test_predict==1)//test an already trained network
    {
        segment_class segment1(0,0,"default_segment");
        vector<nn_core_filtered_data> f_data_vector;
        datapack_structure_defination ds=datapack_analyzer(&data_pack);
        int data_div_max=filter(data_pack,ds,f_data_vector);
        data_pack.data.clear();
        data_pack.labels.clear();
        segment1.set_ds(ds);
        if(train_test_predict==1)
        {
            auto start = high_resolution_clock::now();
            if(critical_variables.id==-1)
            {
                genetic_algorithm ga(iterations,population_size,mutation_percentage,data_div_max);
                ga.ds=ds;
                ga.f_data_vector=&f_data_vector;
                critical_variables=ga.start_genetic_algorithm(batch_size);//control ram usage by lowering this value
            }
            segment1.critical_variable=&critical_variables;
            segment1.set_critical_variable();
            cout<<"\nflatening_fx_enabled: "<<critical_variables.flatening_fx_enabled;
            cout<<"\nzero_weight_remover: "<<critical_variables.zero_weight_remover;
            cout<<"\nextreame_weight_remover: "<<critical_variables.extreame_weight_remover;
            cout<<"\nfp_change_value: "<<critical_variables.fp_change_value;
            cout<<"\nsummation_temp_threshold: "<<critical_variables.summation_temp_thershold;
            cout<<"\nrhs_upper: "<<critical_variables.rhs_upper;
            cout<<"\nrhs_lower: "<<critical_variables.rhs_lower;
            cout<<"\nattributes_per_core: "<<critical_variables.attributes_per_core;
            cout<<"\ndata_division: "<<critical_variables.data_division;
            start_segment_trainer(&segment1,&f_data_vector);
            segment1.clear_segment();
            auto end = high_resolution_clock::now();
            auto duration = duration_cast<microseconds>(end - start);
            cout<<"\n\nTime Taken= "+to_string(duration.count()/pow(10,6))+"\n\n";
        }
        else if(train_test_predict==4)
        {
            bool network_load_status=segment1.load_segment(segment_dir);
            segment1.add_f_data(f_data_vector);
            if(network_load_status)
            {   
                if(segment1.is_network_compatible_with_data())
                {   segment1.data_division=0;chromosome cv;testing_for_each_label(&cv,&segment1,true);}
                else
                {   cout<<"\nERROR!! loaded network is not compatible with loaded dataset.";}
            }
            else
            {   cout<<"\nERROR!!! failed to load network from the network file.";}
        }   
    }
}

void segment_starter(string &file_name_local,int &test_train_predict,string &segment_save_file_name,int &batch_size)
{
    nn_core_data_package_class data_pack;
    if(test_train_predict==1 || test_train_predict==2 || test_train_predict==4)
    {   
        prepare_data(&data_pack,file_name_local,test_train_predict);
        cout<<"\ndata file reading success!!!\n";
    }
    chromosome critical_variables;
    if(test_train_predict==1)
    {   critical_variables=get_critical_variables_from_user(iterations,population_size,mutation_percentage);}
    start_segment(batch_size,test_train_predict,data_pack,segment_save_file_name,critical_variables);
}
