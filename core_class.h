/*
core class handles natural cores
*/

#include<iostream>
#include<vector>
#include<fstream>
#include<sys/stat.h>
#include<string>
#include<algorithm>
#include<math.h>
#include<stdlib.h>
#include<time.h>
#include<thread>
#include<dirent.h>
#include<unistd.h>
#include<sys/ioctl.h>//for the terminal size
#include<chrono>

#include"neuron_and_ann_class.h"

using namespace std::chrono;
using namespace std;
static pthread_mutex_t lock_1,st_add_lock,cdp_add_lock;
inline bool display_iterations=false;//iteration display switch for debugging the code

struct datapack_structure_defination{
    int no_of_labels;
    int no_of_elements_in_each_record;
    vector<float> elements;
    float lower_firing_constrain_rhs;//=92; //60,150
    float upper_not_firing_constrain_rhs;//=10; //10
};

class core_class;
struct converted_data_pack
{
    vector<vector<float>*> firing_data;
    vector<vector<float>*> not_firing_data;
    //vector<float> objective_function_coefficients;
    vector<float> weight_matrix;//ans stored here.
    int firing_neuron_index;
    float firing_label;
    bool corupt_pack=false;
    //ann *network;
    core_class *core;
    int core_no,segment_no;
};
inline vector<converted_data_pack> cdp_vec,cdp_vec_temp;

struct id
{
    bool slack=false,basic=false,z=false,rhs=false,theta=false;
    int id;
};

struct conflict_id
{
    vector<int> id_vec;
    char completion_code;
};

struct simplex_table_cuda
{
    id* c_id;//no_of_columns-rhs-z
    int c_id_size;
    id* r_id; //no_of_rows-z_row
    int r_id_size;
    float* basic_var; //no_of_column-slack_var-z-rhs-theta*no_of_rows-z_row
    int basic_var_size_row,basic_var_size_col;
    float* slack_var; //no_of_columns-basic_var-rhs-theta*no_of_rows-z_row
    int slack_var_size_row,slack_var_size_col;
    double* rhs;//no_of_rows-z_row//actual double
    int rhs_size;
    double* theta;
    //below variables are not used in cuda code, but used only in cpu code
    int core_no,segment_no;
    //int completion_code=0;//0=no solved, 1=solved, 2=conflict found, -1=bad p_row bug, -2= cyclic bug
};
inline vector<simplex_table_cuda*> st_vec,st_vec_temp;
void add_st_cdp(simplex_table_cuda* st,converted_data_pack cdp);
void add_st_cdp_temp(simplex_table_cuda* st1,converted_data_pack cdp1);

void display_st(simplex_table_cuda *st);

vector<conflict_id> simplex_solver();

void free_simplex_table_from_ram(simplex_table_cuda *st);

simplex_table_cuda* generate_simplex_table(converted_data_pack* cdp,datapack_structure_defination ds);

struct buffer
{
    short large_size,large_index,small_size,small_index;
    int *p_row_index;
    int *p_col_index;
    int *p_row_index_small;
    int *p_col_index_small;
};

struct network_structure_defination{
    int no_of_input_neuron=0;
    int no_of_output_neuron=0;
};

class core_class
{
    private:
    bool display_core_events=false;
    //training data pointers
    //core identification information
    string core_name;
    string core_save_file_name="NULL";//provided if core is loaded from a core/network savefile. Not set using constructor
    bool id_lock=false;
    //training information
    network_structure_defination ns;

    void network_structure_modifier();

    void network_analyzer();//it fills up the network_structure_defination ns based on the initialized network structure. 

    void clrscr();
    vector<string> line_breaker(string line);
    
    public:
    int parent_segment_aim=0,parent_segment_no=0;
    int core_no=0,core_aim=0;//this two must be changed using a function so that proper core is loaded
    ann network1;
    datapack_structure_defination ds;//for the data which will be processed by this particular core
    vector<converted_data_pack> c_datapacks;
    vector<nn_core_filtered_data>* f_data_pack;

    void big_c_datapack_handler(vector<converted_data_pack> &cdp);//passing the vector by reference //this function might be a temporary offer //this is for preventing 0:0 bug

    void print_message(string message);
    
    bool load_core(string core_file_dir="");

    void clear_core();//deletes all the data and network present inside the core. This function is not yet implemented.
    
    vector<neuron> propagate(vector<float> input_attributes_value);//need to be implemented

    void load_training_data_into_core(vector<nn_core_filtered_data> *f_data_pack1);

    void save_core(string folder_dir);

    void print_all_path();
    
    network_structure_defination return_ns()
    {   return ns;}
    string return_name()
    {   return core_name;}
    string return_core_savefile_name()
    {   return core_save_file_name;}
    int return_core_no()
    {   return core_no;}
    int return_no_of_input_neuron()
    {   return network1.input_neuron_size();}
    int return_no_of_output_neuron()
    {   return network1.output_neuron_size();}
    datapack_structure_defination return_core_ds()
    {   return ds;}
    int return_core_aim()
    {   return core_aim;}
    int return_no_of_paths()
    {   return network1.path.size();}
    int return_parent_segment_no()
    {   return parent_segment_no;}

    void set_critical_variable(chromosome critical_variable);

    core_class(int core_aim,int core_no,int parent_segment_aim,int parent_segment_no,string core_name,datapack_structure_defination& ds1);
};

void prepare_cdp_and_st_core(core_class *core);
