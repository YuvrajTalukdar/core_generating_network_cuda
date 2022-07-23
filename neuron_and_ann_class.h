/*
Neuron class 
ANN class
*/
#ifndef NEURON_AND_ANN_CLASS_H_
#define NEURON_AND_ANN_CLASS_H_

#include<iostream>
#include<vector>
#include<string>
#include<string.h>
#include<math.h>
#include<pthread.h>
#include"neural_network_core_data_package_class.h"

using namespace std;

class neuron{
    private:
    int id;
    bool input_id,output_id;
    bool fire_status=false;
    float data;
    int label;//this is not the index
    public:
    float firing_point=0;
    
    void set_neuron_identity(int i,bool i_id,bool o_id)
    {
        id=i;
        input_id=i_id;
        output_id=o_id;
    }

    void set_data(float d)
    {
        data=d;
        //cout<<"\ndata="<<d<<endl;
        if(d>firing_point){//i had to change it because of the private solver to -30
            fire_status=true;
        }
    }

    void change_fire_status(bool s)
    {   fire_status=s;}

    void reset_fire_status()
    {   fire_status=false;}

    bool return_fire_status()
    {   return fire_status;}
    
    int get_id()
    {   return id;}
    
    float return_data()
    {   return data;}
};

struct path_struct{
    vector<int> input_neuron_id;
    int output_neuron_id,path_id;
    vector<float> weight_matrix;
};

class ann{
    private:
    //pthread_mutex_t lock;
    //critical variables
    bool flatening_fx_enabled;
    bool zero_weight_remover;
    bool extreame_weight_remover;
    int fp_change_value;
    int summation_temp_thershold;
    //operation variables
    int label_neuron_to_be_fired_id=0;
    bool label_neuron_reset_status=false;

    vector<neuron> input_neurons;
    vector<neuron> output_neurons;    
    
    public:
    vector<path_struct> path;

    void set_critical_variables(chromosome critical_variables)
    {
        flatening_fx_enabled=critical_variables.flatening_fx_enabled;
        zero_weight_remover=critical_variables.zero_weight_remover;
        extreame_weight_remover=critical_variables.extreame_weight_remover;
        fp_change_value=critical_variables.fp_change_value;
        summation_temp_thershold=critical_variables.summation_temp_thershold;
    }

    int return_no_of_paths()
    {   return path.size();}

    void reset_all_output_neurons()
    {
        label_neuron_reset_status=true;
        for(int a=0;a<output_neurons.size();a++)
        {   output_neurons[a].reset_fire_status();}
    }
    
    void enter_data_in_the_network(vector<float> input_matrix) //data is pushed into the input neurons.
    {
        for(int a=0;a<input_neurons.size();a++)
        {   input_neurons[a].set_data(input_matrix[a]);}
    }

    void set_no_of_input_neuron(int n)
    {   
        input_neurons.resize(n);
        for(int a=0;a<input_neurons.size();a++)
        {   input_neurons[a].set_neuron_identity(a,true,false);}
    }

    void set_no_of_output_neuron(int n)
    {   
        output_neurons.resize(n);
        for(int a=0;a<output_neurons.size();a++)
        {   output_neurons[a].set_neuron_identity(a,false,true);}
    }

    void create_new_path(vector<float> weight_matrix,int output_id)//to be called only during training and not during loading
    {
        //pthread_mutex_lock(&lock);
        int zero_count=0,extreame_value=0;
        for(int a=0;a<weight_matrix.size();a++)
        {
            if(weight_matrix[a]==0)
            {   zero_count++;}
            if(abs(weight_matrix[a]>1000))
            {   extreame_value++;}
        }
        if(!extreame_weight_remover)
        {   extreame_value=0;}
        if(!zero_weight_remover)
        {   zero_count=0;}
        if((zero_count<weight_matrix.size()/2) && (extreame_value<2))
        {   
            path_struct new_path;
            int id=path.size();
            if(id==0)
            {   new_path.path_id=0;}
            else
            {
                id=path[id-1].path_id;
                id++;
                new_path.path_id=id;
            }
            //input neuron ids are not required but still included for would be future requeriments
            for(int a=0;a<input_neurons.size();a++)
            {new_path.input_neuron_id.push_back(input_neurons[a].get_id());}
            new_path.output_neuron_id=output_id;
            new_path.weight_matrix.clear();
            for(int a=0;a<weight_matrix.size();a++)
            {   new_path.weight_matrix.push_back(weight_matrix[a]);}

            path.push_back(new_path);
        }
        //pthread_mutex_unlock(&lock);
    }

    int input_neuron_size()
    {   return input_neurons.size();}

    int output_neuron_size()
    {   return output_neurons.size();}

    bool check_network_validity()
    {
        if(input_neuron_size()==0||output_neuron_size()==0)
        {   return false;}
        else
        {   return true;}
    }

    float sigmoid(float summation)
    {
        float exp_value=exp((double)-summation);
        float output=1/(1+exp_value);
        return output;
    }

   vector<neuron> propagate(){
        float summation=0;
        vector<float> summation_vec;
        vector<int> total_no_of_fires_of_paths;
        total_no_of_fires_of_paths.clear();
        //bool sure_positive_found=false,second_neuron_fired_status=false;
        int no_of_neurons_fired=0;
        for(int a=0;a<output_neuron_size();a++)
        {
            summation=0;
            summation_vec.clear();
            output_neurons[a].firing_point=0;
            for(int c=0;c<path.size();c++)
            {
                float summation_temp=0;
                if(path[c].output_neuron_id==a)
                {
                    output_neurons[a].firing_point+=fp_change_value;//50,40
                    for(int d=0;d<path[c].weight_matrix.size();d++)//weight matrix size = input neuron size
                    {
                        summation_temp=summation_temp+input_neurons[d].return_data()*path[c].weight_matrix[d]; //need to be modified...................................
                        //cout<<"i= "<<input_neurons[d].return_data()<<" w= "<<path[c].weight_matrix[d]<<endl;
                    }
                    if(summation_temp>summation_temp_thershold || summation_temp<summation_temp_thershold*-1)//500
                    {
                        summation_temp=0;
                        output_neurons[a].firing_point-=fp_change_value;//50,40
                    }
                    else if(summation_temp>0&&summation_temp<40)
                    {
                        summation_temp=0;
                        output_neurons[a].firing_point-=fp_change_value;//50,40
                    }
                    //cout<<"\nsummation_temp1= "<<summation_temp;
                    if(flatening_fx_enabled)
                    {   summation_temp=((atan(summation_temp)*180/3.1415)/90)*100;}
                    summation_vec.push_back(summation_temp);
                    //cout<<"\nsummation_temp2= "<<summation_temp;
                    summation=summation+summation_temp;
                }
            }
            if(summation-output_neurons[a].firing_point>100000)//100000
            {   summation=0;}
            if(summation>=100000)//100000
            {   summation=0;}
            //cout<<"\nsummation= "<<summation<<" label_neuron_to_be_fired_id= "<<label_neuron_to_be_fired_id<<"a= "<<a;
            //cout<<" firing_point= "<<firing_point;
            output_neurons[a].set_data(summation);
            //counter for df and nf.
            if(output_neurons[a].return_fire_status()==true)
            {   no_of_neurons_fired++;}
        }
        return output_neurons;
    }
};
#endif