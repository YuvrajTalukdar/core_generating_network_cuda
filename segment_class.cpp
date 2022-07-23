#include"segment_class.h"

void segment_class::predict_progress_bar()//need analyzing 
{
    while(predict_progress_bar_numerator<predict_progress_bar_denominator)
    {
        float x=predict_progress_bar_numerator,y=predict_progress_bar_denominator;
        if(y!=0)
        {
            struct winsize w;
            ioctl(0,TIOCGWINSZ,&w);
            float percentage=(x/y)*100;
            clrscr();
            cout<<"\nprogress: ";
            float hl=w.ws_col/2;
            float ratio=100/hl;
            float pl=percentage*hl/100;
            for(int a=0;a<hl;a++)
            {
                if(a<pl)
                {   cout<<"#";}
                else
                {   cout<<".";}
            }
            cout<<"  "<<percentage<<"%";
            cout<<"  "<<predict_progress_bar_numerator<<" out of "<<predict_progress_bar_denominator<<" predictions complete"<<endl<<endl;
            sleep(1);
        }
    }
}

bool segment_class::is_network_compatible_with_data()
{
    int total_input_neuron=0;
    for(int a=0;a<core_vector.size();a++)
    {   total_input_neuron+=core_vector[a]->return_no_of_input_neuron();}
    if(f_data_vector[0].data[0].size()==total_input_neuron && f_data_vector.size()==core_vector[0]->return_no_of_output_neuron())
    {   return true;}
    else
    {   return false;}
}

vector<string> segment_class::line_breaker(string line)
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
bool segment_class::load_segment(string segment_dir)//under construction
{
    string segment_file_name;
    for (const auto & entry : fs::directory_iterator(segment_dir))
    {   
        if(strcasestr(entry.path().filename().c_str(),"segment-")!=NULL)
        {   segment_file_name=entry.path().filename();break;}
    }
    //read segment file
    ifstream file1(segment_dir+"/"+segment_file_name,ios::in);
    int no_of_cores=9;
    int line_count=0;
    vector<string> elements;
    string line;
    try{
        while(file1)
        {
            file1>>line;
            if(line_count==1)//get segment name
            {   
                elements=line_breaker(line);
                segment_save_file_name=elements[1];
            }
            else if(line_count==4)//segment id data
            {
                elements=line_breaker(line);
                segment_aim=stoi(elements[2]);
                segment_no=stoi(elements[3]);
            }
            else if(line_count==5)
            {
                elements=line_breaker(line);
                int state=stoi(elements[1]);
                critical_variable = new chromosome();
                if(state==0)
                {   critical_variable->flatening_fx_enabled=false;}
                else if(state==1)
                {   critical_variable->flatening_fx_enabled=true;}
            }
            else if(line_count==6)
            {
                elements=line_breaker(line);
                critical_variable->summation_temp_thershold=stoi(elements[1]);
            }
            else if(line_count==8)//no of cores
            {
                elements=line_breaker(line);
                no_of_cores=stoi(elements[1]);
            }
            else if(line_count>9)//core id and file information
            {
                elements=line_breaker(line);
                core_class* core=new core_class(stoi(elements[2]),stoi(elements[1]),segment_aim,segment_no,elements[3],ds);
                core->set_critical_variable(*critical_variable);
                core->load_core(segment_dir+"/"+elements[4]);
                core_vector.push_back(core);
            }
            line_count++;
            if(core_vector.size()==no_of_cores)
            {   break;}
        }
    }
    catch(exception e)
    {   
        file1.close();
        return false;
    }
    file1.close();
    if(core_vector.size()==0)
    {   return false;}
    else
    {   
        ds.elements=core_vector[0]->return_core_ds().elements;
        ds.no_of_elements_in_each_record=0;
        for(int a=0;a<core_vector.size();a++)
        {   ds.no_of_elements_in_each_record+=core_vector[a]->return_no_of_input_neuron();}
        ds.no_of_labels=core_vector[0]->return_no_of_output_neuron();
        message="\nSegment Loaded Successfully...";
        print_message();
        return true;
    }
}

void segment_class::checker_df(vector<neuron> &output_neurons)
{
    vector<neuron*> temp_neuron_vect;
    neuron *temp_neuron;
    temp_neuron_vect.clear();
    for(int b=0;b<output_neurons.size();b++)
    {
        //cout<<"\nsummation= "<<output_neurons[b].return_data()<<" label_neuron_to_be_fired_id= "<<label_neuron_to_be_fired_id<<"b= "<<b;
        //cout<<" firing_point= "<<output_neurons[b].return_firing_point();
        if(output_neurons[b].return_fire_status()==true)
        {   temp_neuron_vect.push_back(&output_neurons[b]);}
    }
    //sorting. More optimized sorting algorithm can be used, but if no of output neurons are less than it dose not make a difference.
    for(int b=0;b<temp_neuron_vect.size();b++)
    {
        for(int c=0;c<temp_neuron_vect.size();c++)
        {
            if((temp_neuron_vect[b]->return_data()-temp_neuron_vect[b]->firing_point) > (temp_neuron_vect[c]->return_data()-temp_neuron_vect[c]->firing_point))
            {
                temp_neuron=temp_neuron_vect[b];
                temp_neuron_vect[b]=temp_neuron_vect[c];
                temp_neuron_vect[c]=temp_neuron;
            }
        }
    }
    for(int b=0;b<temp_neuron_vect.size();b++)
    {
        //cout<<"\nsummation= "<<temp_neuron_vect[b]->return_data();
        //cout<<" firing_point= "<<temp_neuron_vect[b]->return_firing_point();
        if(b==0)
        {   temp_neuron_vect[b]->change_fire_status(true);}
        else
        {   temp_neuron_vect[b]->change_fire_status(false);}
    }
}

void segment_class::checker_nf(vector<neuron> &output_neurons)
{
    vector<neuron*> neuron_temp_vect;
    neuron *neuron_temp;
    neuron_temp_vect.clear();
    for(int a=0;a<output_neurons.size();a++)
    {   neuron_temp_vect.push_back(&output_neurons[a]);}
    //sorting
    for(int a=0;a<output_neurons.size();a++)
    {
        for(int b=0;b<output_neurons.size();b++)
        {
            if(neuron_temp_vect[a]->return_data()>neuron_temp_vect[b]->return_data())
            {
                neuron_temp=neuron_temp_vect[a];
                neuron_temp_vect[a]=neuron_temp_vect[b];
                neuron_temp_vect[b]=neuron_temp;
            }
        }
    }
    for(int a=0;a<neuron_temp_vect.size();a++)
    {
        if(neuron_temp_vect[a]->firing_point<100000)
        {   neuron_temp_vect[a]->change_fire_status(true);break;}
    }
}

vector<neuron> segment_class::combine_output_neurons(vector<vector<neuron>> output_neuron_matrix)
{
    vector<neuron> output_neuron;
    output_neuron.resize(output_neuron_matrix[0].size());
    float value_vec[output_neuron.size()]={0};
    //float fp_vec[output_neuron.size()]={0};
    
    for(int a=0;a<output_neuron_matrix.size();a++)
    {
        for(int b=0;b<output_neuron_matrix[a].size();b++)
        {   
            value_vec[b]+=output_neuron_matrix[a][b].return_data();
            output_neuron[b].firing_point+=output_neuron_matrix[a][b].firing_point;
            //fp_vec[b]+=output_neuron_matrix[a][b].firing_point;
        }
    }
    for(int a=0;a<output_neuron.size();a++)
    {   output_neuron[a].set_data(value_vec[a]);}
    return output_neuron;
}

int segment_class::propagate(vector<float> input)//will return the index of 1 fired neuron. This is working for shuttle converted dataset.
{
    int attribute_end_index=0;
    vector<float> trimed_input;
    vector<vector<neuron>> output_neuron_from_each_core;
    for(int a=0;a<core_vector.size();a++)
    {
        trimed_input.clear();
        trimed_input.insert(trimed_input.begin(),input.begin()+attribute_end_index,input.begin()+attribute_end_index+core_vector[a]->return_no_of_input_neuron());
        attribute_end_index=attribute_end_index+core_vector[a]->return_no_of_input_neuron();
        output_neuron_from_each_core.push_back(core_vector[a]->propagate(trimed_input));
    }
    /*cout<<"\n\n";
    for(int a=0;a<output_neuron_from_each_core.size();a++)
    {
        cout<<"\n";
        for(int b=0;b<output_neuron_from_each_core[a].size();b++)
        {
            cout<<"fp="<<output_neuron_from_each_core[a][b].firing_point<<" val="<<output_neuron_from_each_core[a][b].return_data();
        }
    }*/
    vector<neuron> output_neuron=combine_output_neurons(output_neuron_from_each_core);
    output_neuron_from_each_core.clear();
    int no_of_fired_neurons=0;
    for(int a=0;a<output_neuron.size();a++)
    {
        if(output_neuron[a].return_fire_status())
        {   no_of_fired_neurons++;}
    }
    if(no_of_fired_neurons>1)
    {   checker_df(output_neuron);}
    else if(no_of_fired_neurons==0)
    {   checker_nf(output_neuron);}
    int fired_neuron_index,fired_neuron_label;
    int count=0;
    for(int a=0;a<output_neuron.size();a++)
    {
        if(output_neuron[a].return_fire_status())
        {   fired_neuron_index=a;count++;}
    }
    if(count==0)//no fire
    {   fired_neuron_index=-1;}
    else if(count>1)//double fire
    {   fired_neuron_index=-2;}

    return fired_neuron_index;
}

void segment_class::make_prediction_on_user_entered_data()
{
    clrscr();
    int no_of_input_neurons=0;
    for(int a=0;a<core_vector.size();a++)
    {   no_of_input_neurons+=core_vector[a]->return_no_of_input_neuron();}
    char continue1='y';
    vector<float> data_vector;
    float label;
    int fired_neuron_index,correct;
    while(continue1=='y'||continue1=='Y')
    {
        cout<<"\nEnter the "<<no_of_input_neurons<<" digit data: \n";
        for(int a=0;a<no_of_input_neurons;a++)
        {   
            float data;
            cout<<"a"<<a<<"= ";
            cin>>data;
            data_vector.push_back(data);
        }
        fired_neuron_index=propagate(data_vector);
        label=ds.elements[fired_neuron_index];
        cout<<"\nResult = "<<label<<endl;
        data_vector.clear();
        point1:
        cout<<"\nDo you want to make prediction on another data? (y,n)";
        cin>>continue1;
        if(continue1!='N'&&continue1!='n'&&continue1!='y'&&continue1!='Y')
        {
            cout<<"\nWrong option!!!!";
            goto point1;
        }
    }
}

void segment_class::save_segment()//save all the cores and have a segment structure file
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
    string segment_savefile_id="";//core_aim,core_no,day,month,year
    string segment_aim_str=to_string(segment_aim);
    string segment_no_str=to_string(segment_no);
    segment_savefile_id=segment_aim_str+segment_no_str+year_str+month_str+day_str+hr_str+min_str+sec_str;
    segment_save_file_name="segment-"+segment_savefile_id+".csv";
    string folder_dir="./trained_network/segment_"+segment_savefile_id+"/";
    fs::create_directories(folder_dir);
    ofstream file1(folder_dir+segment_save_file_name,ios::out);
    file1<<"FILE_NAME:,\n";
    file1<<"name=,"<<segment_save_file_name<<",\n";
    file1<<"BASIC_SAVEFILE_INFO:,\n";
    file1<<",segment_save_id,segment_aim,segment_no,year,month,day,hour,minute,sec,\n";
    file1<<"segment_save_id,"<<segment_savefile_id<<","<<segment_aim<<","<<segment_no<<","<<year_str<<","<<month_str<<","<<day_str<<","<<hr_str<<","<<min_str<<","<<sec_str<<",\n";
    file1<<"flatening_fx_enabled=,";
    if(critical_variable->flatening_fx_enabled)
    {   file1<<1<<",\n";}
    else
    {   file1<<0<<",\n";}
    file1<<"summation_temp_thershold=,"<<critical_variable->summation_temp_thershold<<",\n";
    file1<<"CORE_INFORMATION:,\n";
    file1<<"no_of_cores=,"<<core_vector.size()<<",\n";
    file1<<"sino,core_no,core_aim,core_name,core_save_file_name,\n";
    for(int a=0;a<core_vector.size();a++)
    {   
        core_vector[a]->save_core(folder_dir);
        file1<<a<<","<<core_vector[a]->return_core_no()<<","<<core_vector[a]->return_core_aim()<<","<<core_vector[a]->return_name()<<","<<core_vector[a]->return_core_savefile_name()<<",\n";
    }
    clear_segment();
    file1.close();
}

int segment_class::index_of_neuron_to_be_fired(int label,vector<float> elements)
{
    int index;
    for(int a=0;a<elements.size();a++)
    {
        if(elements[a]==label)
        {   index=a;break;}
    }
    return index;
}

void testing_for_each_label(chromosome *cv,segment_class *segment,bool display_result)//finds the accuracy of each label
{   
    int correct_each_label[segment->f_data_vector.size()],total_each_label[segment->f_data_vector.size()];
    int correct=0,total=0;
    int fired_neuron_index;
    int df[segment->f_data_vector.size()]={0},nf[segment->f_data_vector.size()]={0};
    bool print_correct_incorrect_data=false;
    nn_core_data_package_class correct_data,incorrect_data;
    int b_start;
    bool test_mode=false;
    if(segment->data_division==0)
    {   test_mode=true;}
    for(int a=0;a<segment->f_data_vector.size();a++)
    {
        correct_each_label[a]=0;
        total_each_label[a]=0;
        if(test_mode)
        {   b_start=0;}
        else
        {   b_start=segment->f_data_vector[a].data.size()/segment->data_division;}
        for(int b=b_start;b<segment->f_data_vector[a].data.size();b++)
        {
            fired_neuron_index=segment->propagate(segment->f_data_vector[a].data[b]);
            if(fired_neuron_index==segment->index_of_neuron_to_be_fired(segment->f_data_vector[a].label,segment->ds.elements))
            {   
                correct_each_label[a]++;
                if(print_correct_incorrect_data)
                {
                    correct_data.data.push_back(segment->f_data_vector[a].data[b]);
                    correct_data.labels.push_back(segment->f_data_vector[a].label);
                }
            }
            else if(fired_neuron_index==-1)
            {   nf[a]++;}
            else if(fired_neuron_index==-2)
            {   df[a]++;}
            else
            {
                if(print_correct_incorrect_data)
                {
                    incorrect_data.data.push_back(segment->f_data_vector[a].data[b]);
                    incorrect_data.labels.push_back(segment->f_data_vector[a].label);
                }
            }
            total_each_label[a]++;
        }
        total+=total_each_label[a];
        correct+=correct_each_label[a];
    }
    if(print_correct_incorrect_data)
    {
        segment->save_data_pack("incorrect_data.csv",incorrect_data);
        segment->save_data_pack("correct_data.csv",correct_data);
    }
    float avg_accuracy=0,accuracy;
    for(int a=0;a<segment->f_data_vector.size();a++)
    {
        accuracy=(((float)correct_each_label[a])/((float)total_each_label[a]))*100;
        if(display_result)
        {
            segment->message="\nAccuracy for label "+to_string(segment->f_data_vector[a].label)+" = "+to_string(accuracy)+"%"+" correct="+to_string(correct_each_label[a])+" total="+to_string(total_each_label[a])+" df="+to_string(df[a])+" nf="+to_string(nf[a]);
            segment->print_message();
        }
        avg_accuracy+=accuracy;
    }
    avg_accuracy=avg_accuracy/segment->f_data_vector.size();
    if(display_result)
    {
        segment->message="\n\nAvg Accuracy= "+to_string(avg_accuracy)+"%"+" correct= "+to_string(correct)+" total= "+to_string(total);
        segment->print_message();
        segment->message="\n\nTotal Accuracy= "+to_string((((float)correct)/((float)total))*100)+"%"+" correct= "+to_string(correct)+" total= "+to_string(total);
        segment->print_message();
    }
    cv->fitness=avg_accuracy;
    //return avg_accuracy;
}

void segment_class::print_prediction(nn_core_data_package_class& data_pack,int train_test_predict)//prints the data and predicted label to a file. Also displays the accuracy.
{
    ofstream file1("prediction_result.csv",ios::out);

    file1<<"data,label,\n";
    float label;
    predict_progress_bar_denominator=data_pack.data.size();
        
    thread* predict_progress_bar_thread;
    if(pds==true)
    {   predict_progress_bar_thread=new thread(&segment_class::predict_progress_bar,this);}
    
    for(int a=0;a<data_pack.data.size();a++)
    {
        for(int b=0;b<data_pack.data[a].size();b++)
        {   file1<<data_pack.data[a][b]<<",";}
        label=ds.elements[propagate(data_pack.data[a])];
        file1<<":"<<label<<",\n";
        predict_progress_bar_numerator++;
    }
    if(pds==true)
    {   predict_progress_bar_thread->join();}
    
    file1.close();
    message="\nPrediction complete, results saved in the file prediction_result.csv\n";
    print_message();
}

void segment_class::split_attributes_for_each_core()//ok tested
{
    //split train data
    int start=0,end;
    vector<float> data;
    for(int a=0;a<f_train_data_split.size();a++)//for each core
    {
        for(int b=0;b<f_data_vector.size();b++)//for each label
        {
            nn_core_filtered_data f_data;
            f_data.label=f_data_vector[b].label;
            int count=f_data_vector[b].data.size()/data_division;
            for(int c=0;c<count;c++)//for each data
            {
                data.clear();
                if(start+no_of_attributes_per_core_balanced<f_data_vector[b].data[c].size())
                {   data.insert(data.end(),f_data_vector[b].data[c].begin()+start,f_data_vector[b].data[c].begin()+(start+no_of_attributes_per_core_balanced));}
                else
                {   data.insert(data.end(),f_data_vector[b].data[c].begin()+start,f_data_vector[b].data[c].end());}
                if(a==f_train_data_split.size()-1 && extra_attributes_in_last_core>0)
                {   data.insert(data.end(),f_data_vector[b].data[c].begin()+(start+no_of_attributes_per_core_balanced),f_data_vector[b].data[c].end());}
                f_data.data.push_back(data);
            }
            f_train_data_split[a].push_back(f_data);
        }
        start+=no_of_attributes_per_core_balanced;
    }
}

void segment_class::create_cores()
{
    int no_of_cores_required;
    int div=ds.no_of_elements_in_each_record/no_of_attributes_per_core_default;
    int rem=ds.no_of_elements_in_each_record%no_of_attributes_per_core_default;
    if(rem>0 && div>0)
    {   no_of_cores_required=div+1;}
    else if(rem==0 && div>0)
    {   no_of_cores_required=div;}
    else if(rem>0 && div==0)
    {   no_of_cores_required=1;}
    no_of_attributes_per_core_balanced=ds.no_of_elements_in_each_record/no_of_cores_required;
    int remaining_attributes=ds.no_of_elements_in_each_record%no_of_attributes_per_core_balanced;
    if(remaining_attributes>0)
    {   
        div=(ds.no_of_elements_in_each_record-no_of_cores_required*no_of_attributes_per_core_balanced)/no_of_attributes_per_core_balanced;
        rem=(ds.no_of_elements_in_each_record-no_of_cores_required*no_of_attributes_per_core_balanced)%no_of_attributes_per_core_balanced;
        if(div>0)
        {   no_of_cores_required+=(div);}
        if(rem>0)
        {
            if(rem>min_no_of_attributes_per_core)
            {   no_of_cores_required++;}
            else
            {   extra_attributes_in_last_core=rem;}
        }
    }

    for(int a=0;a<no_of_cores_required;a++)
    {
        datapack_structure_defination ds1;
        ds1.elements=ds.elements;
        ds1.no_of_labels=ds.no_of_labels;
        ds1.lower_firing_constrain_rhs=ds.lower_firing_constrain_rhs;
        ds1.upper_not_firing_constrain_rhs=ds.upper_not_firing_constrain_rhs;
        core_class* core=new core_class(a,a,segment_aim,segment_no,"core_"+to_string(a),ds1);
        core->set_critical_variable(*critical_variable);
        core_vector.push_back(core);
    }
    f_train_data_split.resize(no_of_cores_required);
}

void segment_class::f_data_viewer(string str,vector<nn_core_filtered_data> f_data)
{
    message.clear();
    message="\n"+str+"\n";
    print_message();
    for(int a=0;a<f_data.size();a++)
    {
        message.clear();
        message="\nlabel= "+to_string(f_data[a].label)+" data_size= "+to_string(f_data[a].data.size());
        print_message();
    }
}

void segment_class::save_data_pack(string name,nn_core_data_package_class data_pack)
{
    ofstream file1(name,ios::out);
    string line="";
    for(int a=0;a<data_pack.data.size();a++)
    {
        for(int b=0;b<data_pack.data[a].size();b++)
        {
            line+=to_string(data_pack.data[a][b]/100);
            line+=",";
        }
        line+=to_string(data_pack.labels[a]/100);
        line+="\n";
        file1<<line;
        line="";
    }
    file1.close();
}

void segment_class::clear_segment()
{
    for(int a=0;a<core_vector.size();a++)
    {   
        core_vector[a]->clear_core();
        delete core_vector[a];
    }
    core_vector.clear();
    f_train_data_split.clear();
}

void prepare_cdp_and_st_segment(segment_class *segment1,vector<nn_core_filtered_data> f_data_vector)
{
    segment1->add_f_data(f_data_vector);
    segment1->set_critical_variable(segment1->critical_variable);
    segment1->f_train_data_split.clear();
    segment1->create_cores();
    segment1->split_attributes_for_each_core();
    vector<thread*> thread_vec;
    for(int a=0;a<segment1->core_vector.size();a++)
    {
        segment1->core_vector[a]->load_training_data_into_core(&segment1->f_train_data_split[a]);
        thread *t=new thread(prepare_cdp_and_st_core,segment1->core_vector[a]);
        thread_vec.push_back(t);
    }
    for(int a=0;a<thread_vec.size();a++)
    {   
        thread_vec[a]->join();
        delete thread_vec[a];
    }
    thread_vec.clear();
}

void segment_class::print_message()
{
    pthread_mutex_lock(&lock_1);
    cout<<message;
    //cout<<"\ncore_no= "<<core_no<<"core_name="<<core_name<<" started...";
    pthread_mutex_unlock(&lock_1);
}

void segment_class::add_f_data(vector<nn_core_filtered_data> f_data_vec)
{
    f_data_vector=f_data_vec;
}

void segment_class::set_critical_variable(chromosome* critical_variable1)
{
   no_of_attributes_per_core_default=critical_variable1->attributes_per_core;
   ds.lower_firing_constrain_rhs=critical_variable1->rhs_lower;
   ds.upper_not_firing_constrain_rhs=critical_variable1->rhs_upper;
   data_division=critical_variable1->data_division;
   critical_variable=critical_variable1;
   //for(int a=0;a<core_vector.size();a++)
   //{    core_vector[a]->set_critical_variable(critical_variable1);}
}

void segment_class::set_critical_variable()
{
   no_of_attributes_per_core_default=critical_variable->attributes_per_core;
   ds.lower_firing_constrain_rhs=critical_variable->rhs_lower;
   ds.upper_not_firing_constrain_rhs=critical_variable->rhs_upper;
   data_division=critical_variable->data_division;
   //for(int a=0;a<core_vector.size();a++)
   //{    core_vector[a]->set_critical_variable(critical_variable1);}
}

segment_class::segment_class(int segment_aim1,int segment_no1,string segment_name1)
{
    if(id_lock==false)
    {
        segment_aim=segment_aim1;
        segment_name=segment_name1;
        segment_no=segment_no1;
        id_lock=true;
    }
    else
    {  
        message.clear();
        message="Failed to set core number and core aim as id_lock=true";
        print_message();
    }
}

void segment_class::print_all_path()
{
    for(int a=0;a<core_vector.size();a++)
    {
        if(core_vector[a]->return_parent_segment_no()==segment_no)
        {
            cout<<"\nsegment_no: "<<segment_no<<" core_no: "<<core_vector[a]->return_core_no()<<" path_size: "<<core_vector[a]->return_no_of_paths();
            //core_vector[a]->print_all_path();
        }
    }
}

void segment_class::clrscr()
{
    cout << "\033[2J\033[1;1H";
    //system("clear");
}

void start_segment_trainer(segment_class *segment,vector<nn_core_filtered_data> *f_data_vector)
{
    prepare_cdp_and_st_segment(segment,*f_data_vector);
    point1:
    vector<conflict_id> conflict_id_vec=simplex_solver();
    for(int a=0;a<conflict_id_vec.size();a++)
    {   handle_completed_table(&conflict_id_vec[a],a);}
    cdp_vec.clear();
    st_vec.clear();
    cdp_vec.insert(cdp_vec.end(),cdp_vec_temp.begin(),cdp_vec_temp.end());
    st_vec.insert(st_vec.end(),st_vec_temp.begin(),st_vec_temp.end());
    cdp_vec_temp.clear();
    st_vec_temp.clear();
    if(cdp_vec.size()!=0)
    {   goto point1;}

    chromosome cv;
    testing_for_each_label(&cv,segment,true);
    segment->save_segment();
}

void free_st_from_ram(simplex_table_cuda *st)
{
    free(st->basic_var);
    free(st->c_id);
    free(st->r_id);
    free(st->rhs);
    free(st->slack_var);
    delete st;
}

void extract_weight_matrix(int index)
{
    vector<float> weight_matrix_raw;
    for(int a=0;a<cdp_vec[index].firing_data[0].size()*2;a++)
    {
        bool found=false;
        for(int b=0;b<st_vec[index]->r_id_size;b++)
        {
            if(st_vec[index]->r_id[b].id==a && st_vec[index]->r_id[b].basic)
            {
                weight_matrix_raw.push_back(st_vec[index]->rhs[b]*st_vec[index]->basic_var[b*st_vec[index]->basic_var_size_col+a]);
                found=true;
                break;
            }
        }
        if(!found)
        {   weight_matrix_raw.push_back(0);}
    }
    for(int a=0;a<cdp_vec[index].firing_data[0].size()*2;a+=2)
    {   cdp_vec[index].weight_matrix.push_back(weight_matrix_raw[a]-weight_matrix_raw[a+1]);}
}

void handle_completed_table(conflict_id *conflict,int index)
{
    if(conflict->completion_code=='1')//table solved successfully
    {
        extract_weight_matrix(index);
        cdp_vec[index].core->network1.create_new_path(cdp_vec[index].weight_matrix,cdp_vec[index].firing_neuron_index);
        /*cout<<"\nweight matrix added to core. weight: ";
        for(int a=0;a<cdp_vec[index].weight_matrix.size();a++)
        {   cout<<cdp_vec[index].weight_matrix[a]<<",";}
        cout<<"\nfd: ";
        for(int a=0;a<cdp_vec[index].firing_data[0].size();a++)
        {   cout<<cdp_vec[index].firing_data[0][a]<<",";}
        cout<<"\nnfd: ";
        for(int a=0;a<cdp_vec[index].not_firing_data[0].size();a++)
        {   cout<<cdp_vec[index].not_firing_data[0][a]<<",";}
        cout<<"\ncore_no_cdp: "<<cdp_vec[index].core_no<<" core_no_st: "<<st_vec[index]->core_no;
        cout<<"\nrhs_lower: "<<segment.critical_variable->rhs_lower<<", rhs_upper: "<<segment.critical_variable->rhs_upper;
        for(int a=0;a<cdp_vec[index].firing_data.size();a++)
        {
            float total=0;
            for(int b=0;b<cdp_vec[index].firing_data[a].size();b++)
            {
                total+=cdp_vec[index].firing_data[a][b]*cdp_vec[index].weight_matrix[b];
            }
            if(total<segment.critical_variable->rhs_lower)
            {   cout<<"\nfailed1!!!  "<<total;}
        }
        for(int a=0;a<cdp_vec[index].not_firing_data.size();a++)
        {
            float total=0;
            for(int b=0;b<cdp_vec[index].not_firing_data[a].size();b++)
            {
                total+=cdp_vec[index].not_firing_data[a][b]*cdp_vec[index].weight_matrix[b];
            }
            if(total>segment.critical_variable->rhs_upper)
            {   cout<<"\nfailed2!!!: "<<total;}
        }
        int gh;cin>>gh;*/
    }
    else//handle conflicts
    {
        converted_data_pack cdp1,cdp2;
        //cout<<"\nfd1: "<<cdp_vec[index].firing_data.size()<<" nfd1: "<<cdp_vec[index].not_firing_data.size();
        //cout<<"\nconflict size: "<<conflict_id_vec[index].id_vec.size()<<" completion_code: "<<conflict_id_vec[index].completion_code;
        //add the conflicting data
        int original_fd_size=cdp_vec[index].firing_data.size();
        if(conflict->completion_code=='2' && !(conflict->id_vec.size()>original_fd_size/2))
        {
            for(int a=0;a<conflict->id_vec.size();a++)
            {   
                //cout<<"\nid: "<<conflict_id_vec[index].id_vec[a];
                cdp1.firing_data.push_back(cdp_vec[index].firing_data[conflict->id_vec[a]]);
            }
            for(int a=0;a<conflict->id_vec.size();a++)
            {   cdp_vec[index].firing_data.erase(cdp_vec[index].firing_data.begin()+conflict->id_vec[a]);}
            //cout<<"\nerasing done";
            cdp1.firing_data.insert(cdp1.firing_data.end(),cdp_vec[index].firing_data.begin(),cdp_vec[index].firing_data.begin()+original_fd_size/2-conflict->id_vec.size());
            cdp2.firing_data.insert(cdp2.firing_data.end(),cdp_vec[index].firing_data.begin()+original_fd_size/2-conflict->id_vec.size(),cdp_vec[index].firing_data.end());
        }
        else
        {
            cdp1.firing_data.insert(cdp1.firing_data.end(),cdp_vec[index].firing_data.begin(),cdp_vec[index].firing_data.begin()+cdp_vec[index].firing_data.size()/2);
            cdp2.firing_data.insert(cdp2.firing_data.end(),cdp_vec[index].firing_data.begin()+cdp_vec[index].firing_data.size()/2,cdp_vec[index].firing_data.end());
        }
        cdp1.not_firing_data.insert(cdp1.not_firing_data.end(),cdp_vec[index].not_firing_data.begin(),cdp_vec[index].not_firing_data.begin()+cdp_vec[index].not_firing_data.size()/2);
        cdp2.not_firing_data.insert(cdp2.not_firing_data.end(),cdp_vec[index].not_firing_data.begin()+cdp_vec[index].not_firing_data.size()/2,cdp_vec[index].not_firing_data.end());
        
        cdp1.core_no=cdp_vec[index].core_no;
        cdp1.firing_label=cdp_vec[index].firing_label;
        cdp1.firing_neuron_index=cdp_vec[index].firing_neuron_index;
        cdp1.segment_no=cdp_vec[index].segment_no;
        cdp1.core=cdp_vec[index].core;

        cdp2.core_no=cdp_vec[index].core_no;
        cdp2.firing_label=cdp_vec[index].firing_label;
        cdp2.firing_neuron_index=cdp_vec[index].firing_neuron_index;
        cdp2.segment_no=cdp_vec[index].segment_no;
        cdp2.core=cdp_vec[index].core;
        
        //cout<<"\nfd2: "<<cdp1.firing_data.size()+cdp2.firing_data.size()<<" nfd2: "<<cdp1.not_firing_data.size()+cdp2.not_firing_data.size();
        //cout<<"\ncheck memory";
        //int gh;cin>>gh;
        if(cdp1.firing_data.size()>1 && cdp1.not_firing_data.size()>2)
        {
            simplex_table_cuda *st1=generate_simplex_table(&cdp1,cdp1.core->ds);
            st1->core_no=cdp1.core_no;
            st1->segment_no=cdp1.segment_no;
            add_st_cdp_temp(st1,cdp1);
        }
        if(cdp2.firing_data.size()>1 && cdp2.not_firing_data.size()>1)
        {
            simplex_table_cuda *st2=generate_simplex_table(&cdp2,cdp2.core->ds);
            st2->core_no=cdp2.core_no;
            st2->segment_no=cdp2.segment_no;
            add_st_cdp_temp(st2,cdp2);
        }
    }
    free_st_from_ram(st_vec[index]);
}
