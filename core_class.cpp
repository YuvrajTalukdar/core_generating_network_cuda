#include"core_class.h"

using namespace std;

void display_st(simplex_table_cuda *st)
{
    time_t now = time(0);
    tm* localtm = localtime(&now);
    int hr,min,sec;
    hr=localtm->tm_hour;
    min=localtm->tm_min;
    sec=localtm->tm_sec;
    string rand=to_string(hr)+"_"+to_string(min)+"_"+to_string(sec);
    fstream file1("simplex_table_"+rand+".csv",ios::out);
    file1<<",";
    for(int a=0;a<st->c_id_size;a++)
    {
        if(st->c_id[a].basic==true)
        {
            file1<<"c"<<st->c_id[a].id<<",";
        }
        else if(st->c_id[a].slack==true)
        {
            file1<<"s"<<st->c_id[a].id<<",";
        }
        else if(st->c_id[a].z==true)
        {
            file1<<"z"<<",";
        }
        else if(st->c_id[a].rhs==true)
        {
            file1<<"rhs"<<",";
        }
        else if(st->c_id[a].theta==true)
        {
            file1<<"theta,";
        }
    }
    file1<<"\n";
    for(int a=0;a<st->r_id_size;a++)
    {
        if(st->r_id[a].basic==true)
        {
            file1<<"c"<<st->r_id[a].id<<",";
        }
        else if(st->r_id[a].slack==true)
        {
            file1<<"s"<<st->r_id[a].id<<",";
        }
        else if(st->r_id[a].z==true)
        {
            file1<<"z"<<",";
        }
        else if(st->r_id[a].rhs==true)
        {
            file1<<"rhs"<<",";
        }


        for(int c=0;c<st->basic_var_size_col;c++)
        {
            file1<<st->basic_var[a*st->basic_var_size_col+c]<<",";
        }
        for(int c=0;c<st->slack_var_size_col;c++)
        {
            file1<<st->slack_var[a*st->slack_var_size_col+c]<<",";
        }
        file1<<","/*<<st.z_col[a]<<","*/<<st->rhs[a]<<","/*<<st->theta[a]<<","*/;

        file1<<"\n";
    }
    /*file1<<"z,";
    for(int a=0;a<st.z_row.size();a++)
    {
        file1<<st.z_row[a]<<",";
    }*/

    file1.close();
}

void free_simplex_table_from_ram(simplex_table_cuda *st)
{
    free(st->basic_var);
    free(st->c_id);
    free(st->r_id);
    free(st->rhs);
    free(st->slack_var);
    free(st->theta);
    delete st;
}

void add_st_cdp(simplex_table_cuda* st,converted_data_pack cdp)
{
    pthread_mutex_lock(&st_add_lock);
    st_vec.push_back(st);
    cdp_vec.push_back(cdp);
    pthread_mutex_unlock(&st_add_lock);
}

void add_st_cdp_temp(simplex_table_cuda* st1,converted_data_pack cdp1)
{
    pthread_mutex_lock(&st_add_lock);
    st_vec_temp.push_back(st1);
    cdp_vec_temp.push_back(cdp1);
    pthread_mutex_unlock(&st_add_lock);
}

vector<neuron> core_class::propagate(vector<float> input_attributes_value)
{
    network1.enter_data_in_the_network(input_attributes_value);
    return network1.propagate();
}

void core_class::save_core(string folder_dir)
{
    time_t theTime = time(NULL);
    struct tm *aTime = localtime(&theTime);
    int day = aTime->tm_mday;
    string day_str= to_string(day);
    if(day<10)
    {   day_str="0"+day_str;}
    int month = aTime->tm_mon + 1; // Month is 0 - 11, add 1 to get a jan-dec 1-12 concept
    string month_str=to_string(month);
    if(month<10)
    {   month_str="0"+month_str;}
    int year = aTime->tm_year + 1900; // Year is # years since 1900
    string year_str=to_string(year);
    time_t now = time(0);
    // Convert now to tm struct for local timezone
    tm* localtm = localtime(&now);
    int hr,min,sec;
    hr=localtm->tm_hour;
    min=localtm->tm_min;
    sec=localtm->tm_sec;
    string hr_str=to_string(hr),min_str=to_string(min),sec_str=to_string(sec);
    if(hr<10)
    {   hr_str="0"+hr_str;}
    if(min<10)
    {   min_str="0"+min_str;}
    if(sec<10)
    {   sec_str="0"+sec_str;}
    string core_savefile_id="";//core_aim,core_no,day,month,year
    string core_aim_str=to_string(core_aim);
    string core_no_str=to_string(core_no);
    core_savefile_id=core_aim_str+core_no_str+year_str+month_str+day_str+hr_str+min_str+sec_str;
    core_save_file_name="core-"+core_savefile_id+".csv";
    ofstream file1(folder_dir+core_save_file_name,ios::out);
    file1<<"FILE_NAME:,\n";
    file1<<"name=,"<<core_save_file_name<<",\n";
    file1<<"BASIC_SAVEFILE_INFO:,\n";
    file1<<",core_save_id,core_aim,core_no,year,month,day,hour,minute,sec,\n";
    file1<<"core_save_id,"<<core_savefile_id<<","<<core_aim<<","<<core_no<<","<<year_str<<","<<month_str<<","<<day_str<<","<<hr_str<<","<<min_str<<","<<sec_str<<",\n";
    file1<<"BASIC_NETWORK_INFO:,";
    file1<<"net_info_category,no_of_input_neuron,no_of_output_neuron,\n";
    file1<<",net_info_category,"<<network1.input_neuron_size()<<","<<network1.output_neuron_size()<<",\n";
    file1<<"DATA_LABEL_AND_OUTPUT_NEURON_INDEX:,\n";
    file1<<"[data_label~output_neuron_index],";
    for(int a=0;a<ds.elements.size();a++)
    {
        file1<<ds.elements[a]<<"~"<<a<<",";
    }
    file1<<"\n";
    file1<<"PATH_INFO:,\n";
    file1<<"no_of_path=,"<<network1.return_no_of_paths()<<",\n";
    file1<<"path_id,";
    if(network1.path.size()>0)
    {
        for(int a=0;a<network1.path[0].weight_matrix.size();a++)
        {
            int weight_index=a;
            string weight_index_str=to_string(weight_index);
            weight_index_str="w"+weight_index_str;
            file1<<weight_index_str<<",";
        }
    }
    file1<<"input_neuron_id,output_neuron_id\n";
    for(int a=0;a<network1.path.size();a++)
    {
        file1<<network1.path[a].path_id<<",";
        for(int b=0;b<network1.path[a].weight_matrix.size();b++)
        {
            file1<<network1.path[a].weight_matrix[b]<<",";
        }
        file1<<"[|";
        for(int b=0;b<network1.path[a].input_neuron_id.size();b++)
        {
            file1<<network1.path[a].input_neuron_id[b]<<"|";
        }
        file1<<"],";
        file1<<network1.path[a].output_neuron_id<<",\n";
    }
    file1.close();
}

vector<string> core_class::line_breaker(string line)
{
    vector<string> elements;
    string word="";
    for(int a=0;a<line.size();a++)
    {
        if(line.at(a)==',')
        {
            elements.push_back(word);
            word="";
            continue;
        }
        word+=(line.at(a));
    }
    return elements;
}
bool core_class::load_core(string core_file_dir)
{
    ifstream file1(core_file_dir,ios::in);
    string line="";
    int line_count=0;
    vector<string> elements;
    int no_of_path=10;
    try{
        while(file1)
        {
            file1>>line;
            if(line_count==1)
            {
                elements=line_breaker(line);
                core_save_file_name=elements[1];
            }
            else if(line_count==4)
            {
                elements=line_breaker(line);
                core_aim=stoi(elements[2]);
                core_no=stoi(elements[3]);
            }
            else if(line_count==6)
            {
                elements=line_breaker(line);
                network1.set_no_of_input_neuron(stoi(elements[2]));
                network1.set_no_of_output_neuron(stoi(elements[3]));
            }
            else if(line_count==8)
            {
                elements=line_breaker(line);
                ds.elements.resize(elements.size()-1);
                string data_label_str="",output_neuron_index_str="";
                for(int a=1;a<elements.size();a++)
                {
                    int symbol_index;
                    for(int b=0;b<elements[a].size();b++)
                    {
                        if(elements[a][b]=='~')
                        {   symbol_index=b;break;}
                    }
                    data_label_str="";
                    data_label_str.insert(data_label_str.begin(),elements[a].begin(),elements[a].begin()+symbol_index);
                    output_neuron_index_str="";
                    output_neuron_index_str.insert(output_neuron_index_str.begin(),elements[a].begin()+symbol_index+1,elements[a].end());
                    ds.elements[stoi(output_neuron_index_str)]=stoi(data_label_str);
                }
            }
            else if(line_count==10)
            {
                elements=line_breaker(line);
                no_of_path=stoi(elements[1]);
            }
            else if(line_count>11)
            {
                elements=line_breaker(line);
                path_struct path;
                path.output_neuron_id;
                for(int a=0;a<network1.input_neuron_size();a++)
                {   
                    path.input_neuron_id.push_back(a);
                    path.weight_matrix.push_back(stof(elements[a+1]));
                }
                path.output_neuron_id=stoi(elements[elements.size()-1]);
                network1.path.push_back(path);
            }
            line_count++;
            if(network1.path.size()==no_of_path)
            {   break;}
        }
    }
    catch(exception q)
    {   
        file1.close();
        return false;
    }
    file1.close();
    network_analyzer();//initialization of ns
    print_message("\nCore "+to_string(core_no)+" Loaded Successfully...");
    return true;
}

void core_class::network_analyzer()
{
    ns.no_of_input_neuron=network1.input_neuron_size();
    ns.no_of_output_neuron=network1.output_neuron_size();
}

void core_class::big_c_datapack_handler(vector<converted_data_pack> &cdp)//passing the vector by reference //this function might be a temporary offer //this is for preventing 0:0 bug
{
    int limit=50;//the fd and nfd size never goes above limit*2
    converted_data_pack cdp_temp1,cdp_temp2;
    vector<converted_data_pack> cdp_vect_temp;
    for(int a=cdp.size()-1;a>=0;a--)
    {
        if(cdp[a].firing_data.size()>limit)
        {
            cdp_temp1=cdp[a];
            cdp.erase(cdp.begin()+a);
            int begin=0,end=0;
            bool end_reached=false;
            while(end_reached==false)
            {
                cdp_temp2.firing_data.clear();
                cdp_temp2.not_firing_data.clear();
                begin=end;
                end=begin+limit;
                if(end>=cdp_temp1.firing_data.size())
                {   end=cdp_temp1.firing_data.size();end_reached=true;}
                cdp_temp2.firing_data.insert(cdp_temp2.firing_data.end(),cdp_temp1.firing_data.begin()+begin,cdp_temp1.firing_data.begin()+end);
                if(end_reached==true)
                {   end=cdp_temp1.not_firing_data.size();}
                else if(end>=cdp_temp1.not_firing_data.size())
                {
                    int end2=cdp_temp1.firing_data.size();
                    cdp_temp2.firing_data.insert(cdp_temp2.firing_data.end(),cdp_temp1.firing_data.begin()+end,cdp_temp1.firing_data.begin()+end2);
                    end=cdp_temp1.not_firing_data.size();
                    end_reached=true;
                }
                cdp_temp2.not_firing_data.insert(cdp_temp2.not_firing_data.end(),cdp_temp1.not_firing_data.begin()+begin,cdp_temp1.not_firing_data.begin()+end);
                cdp_temp2.firing_neuron_index=cdp_temp1.firing_neuron_index;
                cdp_vect_temp.push_back(cdp_temp2);
            }
            cdp_temp1.firing_data.clear();
            cdp_temp1.not_firing_data.clear();
        }
    }
    cdp.insert(cdp.end(),cdp_vect_temp.begin(),cdp_vect_temp.end());
    print_message("\n\ncdp size after erasing in big data handler = "+to_string(cdp.size()));
    print_message(", cdp_vect_temp size= "+to_string(cdp_vect_temp.size()));
    cdp_vect_temp.clear();
    
    for(int a=0;a<cdp.size();a++)
    {
        int difference=cdp[a].firing_data.size()-cdp[a].not_firing_data.size();
        if(abs(difference)>10)
        {
            cdp_temp1.firing_data.clear();
            cdp_temp1.not_firing_data.clear();
            cdp_temp2.firing_data.clear();
            cdp_temp2.not_firing_data.clear();
            cdp_temp1=cdp[a];
            cdp.erase(cdp.begin()+a);
            while(abs(difference)>10)
            {
                if(difference<0)
                {
                    limit=cdp_temp1.firing_data.size();
                    cdp_temp2.firing_data=cdp_temp1.firing_data;
                    cdp_temp2.not_firing_data.insert(cdp_temp2.not_firing_data.end(),cdp_temp1.not_firing_data.begin()+abs(difference),cdp_temp1.not_firing_data.end());
                    cdp_temp2.firing_neuron_index=cdp_temp1.firing_neuron_index;
                    cdp_temp1.not_firing_data.erase(cdp_temp1.not_firing_data.begin()+abs(difference),cdp_temp1.not_firing_data.end());
                    cdp_vect_temp.push_back(cdp_temp2);
                }
                else if(difference>0)
                {
                    limit=cdp_temp1.not_firing_data.size();
                    cdp_temp2.not_firing_data=cdp_temp1.not_firing_data;
                    cdp_temp2.firing_data.insert(cdp_temp2.firing_data.end(),cdp_temp1.firing_data.begin()+abs(difference),cdp_temp1.firing_data.end());
                    cdp_temp2.firing_neuron_index=cdp_temp1.firing_neuron_index;
                    cdp_temp1.firing_data.erase(cdp_temp1.firing_data.begin()+abs(difference),cdp_temp1.firing_data.end());
                    cdp_vect_temp.push_back(cdp_temp2);
                }
                difference=cdp_temp1.firing_data.size()-cdp_temp1.not_firing_data.size();
                cdp_temp2.firing_data.clear();
                cdp_temp2.not_firing_data.clear();
            }
            cdp.push_back(cdp_temp1);
        }
    }
    cdp.insert(cdp.end(),cdp_vect_temp.begin(),cdp_vect_temp.end());
    //print_message("\ncdp size after stabilizing extreme ratios = "+to_string(cdp.size()));
    //print_message(", cdp_vect_temp size= "+to_string(cdp_vect_temp.size()));
}

void core_class::load_training_data_into_core(vector<nn_core_filtered_data> *f_data_pack1)
{
    f_data_pack=f_data_pack1;
    ds.no_of_elements_in_each_record=f_data_pack->at(0).data[0].size();
    network_structure_modifier();
}

void prepare_cdp_and_st_core(core_class *core)
{
    converted_data_pack c_datapack;
    //ratio maintance and packing data in c_datapacks.
    int sum_total_training_data=0;
    for(int a=0;a<core->f_data_pack->size();a++)
    {   sum_total_training_data=sum_total_training_data+core->f_data_pack->at(a).data.size();}
    core->print_message("\nsize of training data set= "+to_string(sum_total_training_data)+"\n");
    core->c_datapacks.clear(); //for asured cleaniness
    for(int a=0;a<core->f_data_pack->size();a++)
    {
        core->print_message("packing data for label= "+to_string(core->f_data_pack->at(a).label)+"\n");
        //determining the c_data_pack critical info
        int sum_total_not_firing_data=sum_total_training_data-core->f_data_pack->at(a).data.size();
        int no_of_c_data_packs_needed=0,no_of_not_firing_data_in_each_pack=0,no_of_firing_data_in_each_pack=0;
        int additional_firing_data_in_the_last_datapack=0,additional_not_firing_data_in_the_last_datapack=0;
        if(sum_total_not_firing_data>=core->f_data_pack->at(a).data.size())//for not firing data > firing data
        {
            while(sum_total_not_firing_data>=core->f_data_pack->at(a).data.size())
            {
                //cout<<"\ncheck4="<<f_data_pack[a].data.size()<<" "<<sum_total_not_firing_data;
                sum_total_not_firing_data=sum_total_not_firing_data-core->f_data_pack->at(a).data.size();//cout<<"check2";
                no_of_c_data_packs_needed++;
            }
            int rem1=sum_total_not_firing_data;
            no_of_firing_data_in_each_pack=core->f_data_pack->at(a).data.size();
            no_of_not_firing_data_in_each_pack=core->f_data_pack->at(a).data.size()+rem1/no_of_c_data_packs_needed;
            additional_not_firing_data_in_the_last_datapack=rem1%no_of_c_data_packs_needed;
        }
        else if(sum_total_not_firing_data<core->f_data_pack->at(a).data.size()) //for firing data more than not firing data
        {
            int sum_total_firing_data=core->f_data_pack->at(a).data.size();
            while(sum_total_firing_data>=sum_total_not_firing_data)
            {
                sum_total_firing_data=sum_total_firing_data-sum_total_not_firing_data;//cout<<"check3";
                no_of_c_data_packs_needed++;
            }
            int rem1=sum_total_firing_data;
            no_of_not_firing_data_in_each_pack=sum_total_not_firing_data;
            no_of_firing_data_in_each_pack=sum_total_not_firing_data+rem1/no_of_c_data_packs_needed;
            additional_firing_data_in_the_last_datapack=rem1%no_of_c_data_packs_needed;
        }
        
        //packaging the data
        if(no_of_firing_data_in_each_pack==core->f_data_pack->at(a).data.size())//this means firing data < not firing data
        {
            int no_of_packages_created=0;
            int initial_value=0,final_value=0;
            vector<vector<float>*> not_firing_data_temp;
            not_firing_data_temp.clear();
            //copying all the not firing data in not_firing_data_temp
            for(int b=0;b<core->f_data_pack->size();b++)
            {
                if(b!=a)
                {
                    for(int c=0;c<core->f_data_pack->at(b).data.size();c++)
                    {   not_firing_data_temp.push_back(&core->f_data_pack->at(b).data[c]);}
                }
            }
            while(no_of_packages_created!=no_of_c_data_packs_needed)
            {
                //clearing the buffers
                c_datapack.firing_data.clear();
                c_datapack.not_firing_data.clear();
                //c_datapack.objective_function_coefficients.clear();
                c_datapack.weight_matrix.clear();
                //packing the firing data
                for(int b=0;b<core->f_data_pack->at(a).data.size();b++)
                {   c_datapack.firing_data.push_back(&core->f_data_pack->at(a).data[b]);}
                //packing not firing data
                initial_value=final_value;
                final_value=final_value+no_of_not_firing_data_in_each_pack;
                if(no_of_packages_created==no_of_c_data_packs_needed-1)//for the last package
                {   final_value=final_value+additional_not_firing_data_in_the_last_datapack;}
                //cout<<"\nnot_firing_data_temp size= "<<not_firing_data_temp.size();
                //cout<<"\nfinal_value= "<<final_value<<" initial_value= "<<initial_value;
                //cout<<"\nadditional_not_firing_data_in_the_last_datapack= "<<additional_not_firing_data_in_the_last_datapack;
                for(int b=initial_value;b<final_value;b++)
                {
                    c_datapack.not_firing_data.push_back(not_firing_data_temp[b]);
                }
                //setting up the label and output neuron index
                c_datapack.firing_neuron_index=a;
                //pushing the c_datapack in c_datapacks vector
                core->c_datapacks.push_back(c_datapack);
                no_of_packages_created++;
            }
        }
        else if(no_of_not_firing_data_in_each_pack==sum_total_not_firing_data)//this means firing data > not firing data
        {
            int no_of_packages_created=0;
            int initial_value=0,final_value=0;
            while(no_of_packages_created!=no_of_c_data_packs_needed)
            {
                //clearing the buffers
                c_datapack.firing_data.clear();
                c_datapack.not_firing_data.clear();
                //c_datapack.objective_function_coefficients.clear();
                c_datapack.weight_matrix.clear();
                //packing the firing data
                initial_value=final_value;
                final_value=final_value+no_of_firing_data_in_each_pack;
                if(no_of_packages_created==no_of_c_data_packs_needed-1)
                {   final_value=final_value+additional_firing_data_in_the_last_datapack;}
                for(int b=initial_value;b<final_value;b++)
                {
                    c_datapack.firing_data.push_back(&core->f_data_pack->at(a).data[b]);
                }
                //packing the not firing data
                for(int b=0;b<core->f_data_pack->size();b++)
                {
                    if(b!=a)
                    {
                        for(int c=0;c<core->f_data_pack->at(b).data.size();c++)
                        {   c_datapack.not_firing_data.push_back(&core->f_data_pack->at(b).data[c]);}
                    }
                }
                //setting up the label and output neuron index
                c_datapack.firing_neuron_index=a;
                core->c_datapacks.push_back(c_datapack);
                no_of_packages_created++;
            }
        }
    }
    //core->f_data_pack->clear();//memory_optimization3
    core->print_message("finished packaging data in c_datapacks.");
    core->print_message("\ntotal no of c_data_packs= "+to_string(core->c_datapacks.size()));
    /*cout<<"\nsize1: "<<c_datapacks.size();
    for(int a=0;a<c_datapacks.size();a++)
    {
        cout<<"\na: "<<a<<" fd: "<<c_datapacks[a].firing_data.size()<<" nfd: "<<c_datapacks[a].not_firing_data.size();
    }*/
    core->big_c_datapack_handler(core->c_datapacks);//for handling c_datapack with huge data which may create full conlflict senarios.
    core->print_message("\ntotal no of c_data_packs after big c_datapacks handling= "+to_string(core->c_datapacks.size()));
    
    for(int a=0;a<core->c_datapacks.size();a++)
    {   
        //cout<<"\nfd_size: "<<c_datapacks[a].firing_data.size()<<" nfd_size: "<<c_datapacks[a].not_firing_data.size();
        simplex_table_cuda *st=generate_simplex_table(&core->c_datapacks[a],core->ds); 
        st->core_no=core->core_no;
        st->segment_no=core->parent_segment_no;
        core->c_datapacks[a].core=core;
        add_st_cdp(st,core->c_datapacks[a]);
    }
    core->c_datapacks.clear();
}

void core_class::network_structure_modifier()
{
    try{
        if(ns.no_of_input_neuron>ds.no_of_elements_in_each_record)
        {
            throw("network has more neuron than required by the data");//needs working here. UNDER CONSTRUCTION.
        }
        else
        {
            network1.set_no_of_input_neuron(ds.no_of_elements_in_each_record);
            network1.set_no_of_output_neuron(ds.no_of_labels);
        }
    }
    catch(string s)
    {   cout<<s<<endl;}
}

void core_class::print_message(string message)
{
    if(display_core_events)
    {
        pthread_mutex_lock(&lock_1);
        cout<<message;
        pthread_mutex_unlock(&lock_1);
    }
}

void core_class::set_critical_variable(chromosome critical_variable)
{   
    network1.set_critical_variables(critical_variable);
    network1.path.clear();
    ds.lower_firing_constrain_rhs=critical_variable.rhs_lower;
    ds.upper_not_firing_constrain_rhs=critical_variable.rhs_upper;
}

void core_class::clear_core()
{
    c_datapacks.clear();
}

void core_class::print_all_path()
{
    cout<<"\ncore_no: "<<core_no<<": ";
    for(int a=0;a<network1.path.size();a++)
    {
        for(int b=0;b<network1.path[a].weight_matrix.size();b++)
        {   cout<<network1.path[a].weight_matrix[b]<<",";}
        cout<<"\n";
    }
}

core_class::core_class(
    int core_aim1,
    int core_no1,
    int parent_segment_aim1,
    int parent_segment_no1,
    string core_name1,
    datapack_structure_defination& ds1)
{
    if(id_lock==false)
    {
        core_aim=core_aim1;
        core_no=core_no1;
        parent_segment_aim=parent_segment_aim1;
        parent_segment_no=parent_segment_no1;
        core_name=core_name1;
        id_lock=true;
        ds=ds1;
    }
    else
    {
        print_message("Failed to set core number and core aim as id_lock=true");
    }
}

simplex_table_cuda* generate_simplex_table(converted_data_pack* cdp,datapack_structure_defination ds)
{
    simplex_table_cuda *st=new simplex_table_cuda();

    st->c_id_size=cdp->firing_data[0]->size()*2+cdp->firing_data.size()+cdp->not_firing_data.size()+3;
    st->c_id=(id*)malloc(sizeof(id)*st->c_id_size);
    for(int a=0;a<cdp->firing_data[0]->size()*2;a++)
    {
        id temp_id;
        temp_id.basic=true;
        temp_id.slack=false;
        temp_id.z=false;
        temp_id.rhs=false;
        temp_id.theta=false;
        temp_id.id=a;
        st->c_id[a]=temp_id;
    }
    int slack_id=cdp->firing_data[0]->size()*2;
    for(int a=0;a<(cdp->firing_data.size()+cdp->not_firing_data.size());a++)
    {
        id temp_id;
        temp_id.slack=true;
        temp_id.basic=false;
        temp_id.z=false;
        temp_id.theta=false;
        temp_id.rhs=false;
        temp_id.id=slack_id;
        st->c_id[slack_id]=temp_id;
        slack_id++;
    }
    id temp_id;
    temp_id.slack=false;
    temp_id.basic=false;
    temp_id.rhs=false;
    temp_id.z=true;
    temp_id.theta=false;
    temp_id.id=slack_id;
    st->c_id[slack_id]=temp_id;
    slack_id++;

    id temp_id2;
    temp_id2.slack=false;
    temp_id2.basic=false;
    temp_id2.rhs=true;
    temp_id2.z=false;
    temp_id2.theta=false;
    temp_id2.id=slack_id;
    st->c_id[slack_id]=temp_id2;

    temp_id.slack=false;
    temp_id.basic=false;
    temp_id.rhs=false;
    temp_id.z=false;
    temp_id.theta=true;
    temp_id.id=slack_id;
    st->c_id[slack_id+1]=temp_id2;

    st->r_id_size=cdp->firing_data.size()+cdp->not_firing_data.size();
    st->r_id=(id*)calloc(st->r_id_size,sizeof(id));
    int b=0;
    for(int a=0;a<st->c_id_size;a++)
    {
        if(st->c_id[a].slack==true)//check if square bracket works
        {
            st->r_id[b]=(st->c_id[a]);
            b++;
        }
    }

    st->rhs_size=cdp->firing_data.size()+cdp->not_firing_data.size();
    st->rhs=(double*)malloc(sizeof(double)*st->rhs_size);

    st->basic_var_size_row=cdp->firing_data.size()+cdp->not_firing_data.size();
    st->basic_var_size_col=cdp->firing_data[0]->size()*2;
    st->basic_var=(float*)malloc(sizeof(float)*st->basic_var_size_col*st->basic_var_size_row);

    st->slack_var_size_row=st->basic_var_size_row;
    st->slack_var_size_col=st->r_id_size;
    st->slack_var=(float*)malloc(sizeof(float)*st->slack_var_size_col*st->slack_var_size_row);

    for(int a=0;a<cdp->firing_data.size();a++)
    {
        //entering basic variable data
        for(int b=0;b<cdp->firing_data[a]->size();b++)
        {   
            st->basic_var[a*cdp->firing_data[a]->size()*2+b*2]=cdp->firing_data[a]->at(b);
            st->basic_var[a*cdp->firing_data[a]->size()*2+b*2+1]=cdp->firing_data[a]->at(b)*-1;
        }
        //entering slack var data
        for(int b=0;b<st->r_id_size;b++)
        {
            if(b==a)
            {   st->slack_var[a*st->slack_var_size_row+b]=-1;}
            else
            {   st->slack_var[a*st->basic_var_size_row+b]=0;}
        }
        st->rhs[a]=(ds.lower_firing_constrain_rhs); //modification needs to be done here
    }
    int x1=cdp->firing_data.size();
    for(int a=0;a<cdp->not_firing_data.size();a++)
    {
        //entering basic variable data
        for(int b=0;b<cdp->not_firing_data[a]->size();b++)
        {   
            st->basic_var[x1*cdp->not_firing_data[a]->size()*2+b*2]=cdp->not_firing_data[a]->at(b);
            st->basic_var[x1*cdp->not_firing_data[a]->size()*2+b*2+1]=cdp->not_firing_data[a]->at(b)*-1;
        }
        //entering slack var data
        for(int b=0;b<st->r_id_size;b++)
        {
            if((b)==(x1))
            {   st->slack_var[(x1)*st->slack_var_size_row+b]=1;}
            else
            {   st->slack_var[(x1)*st->slack_var_size_row+b]=0;}
        }
        st->rhs[x1]=(ds.upper_not_firing_constrain_rhs); //modification needs to be done here
        x1++;
    }
    return st;
}

void core_class::clrscr()
{
    cout << "\033[2J\033[1;1H";
    //system("clear");
}