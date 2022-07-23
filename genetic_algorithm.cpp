#include"segment_class.h"

int genetic_algorithm::get_random_number(int min,int max)
{
    random_device dev;
    mt19937 rng(dev());
    uniform_int_distribution<std::mt19937::result_type> dist6(min,max); // distribution in range [1, 6]
    return dist6(rng);
}

bool genetic_algorithm::get_random_bool()
{
    int num=get_random_number(0,1);
    if(num==0)
    {   return false;}
    else
    {   return true;}
}

bool genetic_algorithm::comparator(chromosome c1,chromosome c2)
{
    if(c1.fitness>c2.fitness)
    {   return true;}
    else
    {   return false;}
}

void genetic_algorithm::mutation(vector<chromosome>& population)
{
    vector<int> index_temp;
    bool found;
    for(int a=0;a<population.size();a++)
    {
        index_temp.clear();
        for(int b=0;b<no_of_genes_to_mutate;b++)
        {
            point1:
            int index=get_random_number(0,7);
            found=false;
            for(int c=0;c<index_temp.size();c++)
            {
                if(index_temp[c]==index)
                {   found=true;break;}
            }
            if(found)
            {   goto point1;}
            switch (index)
            {
                case 0:
                population[a].flatening_fx_enabled=!population[a].flatening_fx_enabled;
                break;
                case 1:
                population[a].extreame_weight_remover=!population[a].extreame_weight_remover;
                break;
                case 2:
                population[a].zero_weight_remover=!population[a].zero_weight_remover;
                break;
                case 3:
                population[a].fp_change_value=get_random_number(fp_change_value_min,fp_change_value_max);
                break;
                case 4:
                population[a].summation_temp_thershold=get_random_number(summation_temp_thershold_min,summation_temp_thershold_max);
                break;
                case 5:
                population[a].rhs_upper=get_random_number(rhs_upper_min,rhs_upper_max);
                break;
                case 6:
                population[a].rhs_lower=get_random_number(rhs_lower_min,rhs_lower_max);
                break;
                case 7:
                population[a].attributes_per_core=get_random_number(attributes_per_core_min,attributes_per_core_max);
                break;
            }
            index_temp.push_back(index);
        }
    }
}

void genetic_algorithm::chromosome_data_transfer(int crossover_index,bool before,chromosome& source,chromosome& destination)
{
    if(before)
    {
        switch (crossover_index)
        {
            case 0:
            destination.flatening_fx_enabled=source.flatening_fx_enabled;
            break;
            case 1:
            destination.flatening_fx_enabled=source.flatening_fx_enabled;
            destination.extreame_weight_remover=source.extreame_weight_remover;
            break;
            case 2:
            destination.flatening_fx_enabled=source.flatening_fx_enabled;
            destination.extreame_weight_remover=source.extreame_weight_remover;
            destination.zero_weight_remover=source.zero_weight_remover;
            break;
            case 3:
            destination.flatening_fx_enabled=source.flatening_fx_enabled;
            destination.extreame_weight_remover=source.extreame_weight_remover;
            destination.zero_weight_remover=source.zero_weight_remover;
            destination.fp_change_value=source.fp_change_value;
            break;
            case 4:
            destination.flatening_fx_enabled=source.flatening_fx_enabled;
            destination.extreame_weight_remover=source.extreame_weight_remover;
            destination.zero_weight_remover=source.zero_weight_remover;
            destination.fp_change_value=source.fp_change_value;
            destination.summation_temp_thershold=source.summation_temp_thershold;
            break;
            case 5:
            destination.flatening_fx_enabled=source.flatening_fx_enabled;
            destination.extreame_weight_remover=source.extreame_weight_remover;
            destination.zero_weight_remover=source.zero_weight_remover;
            destination.fp_change_value=source.fp_change_value;
            destination.summation_temp_thershold=source.fp_change_value;
            destination.rhs_upper=source.rhs_upper;
            break;
            case 6:
            destination.flatening_fx_enabled=source.flatening_fx_enabled;
            destination.extreame_weight_remover=source.extreame_weight_remover;
            destination.zero_weight_remover=source.zero_weight_remover;
            destination.fp_change_value=source.fp_change_value;
            destination.summation_temp_thershold=source.fp_change_value;
            destination.rhs_upper=source.rhs_upper;
            destination.rhs_lower=source.rhs_lower;
            break;
            case 7:
            destination.flatening_fx_enabled=source.flatening_fx_enabled;
            destination.extreame_weight_remover=source.extreame_weight_remover;
            destination.zero_weight_remover=source.zero_weight_remover;
            destination.fp_change_value=source.fp_change_value;
            destination.summation_temp_thershold=source.fp_change_value;
            destination.rhs_upper=source.rhs_upper;
            destination.rhs_lower=source.rhs_lower;
            destination.attributes_per_core=source.attributes_per_core;
            break;
        }
    }
    else
    {
        switch (crossover_index)
        {
            case 0:
            destination.extreame_weight_remover=source.extreame_weight_remover;
            destination.zero_weight_remover=source.zero_weight_remover;
            destination.fp_change_value=source.fp_change_value;
            destination.summation_temp_thershold=source.fp_change_value;
            destination.rhs_upper=source.rhs_upper;
            destination.rhs_lower=source.rhs_lower;
            destination.attributes_per_core=source.attributes_per_core;
            destination.data_division=source.data_division;
            break;
            case 1:
            destination.zero_weight_remover=source.zero_weight_remover;
            destination.fp_change_value=source.fp_change_value;
            destination.summation_temp_thershold=source.fp_change_value;
            destination.rhs_upper=source.rhs_upper;
            destination.rhs_lower=source.rhs_lower;
            destination.attributes_per_core=source.attributes_per_core;
            destination.data_division=source.data_division;
            break;
            case 2:
            destination.fp_change_value=source.fp_change_value;
            destination.summation_temp_thershold=source.fp_change_value;
            destination.rhs_upper=source.rhs_upper;
            destination.rhs_lower=source.rhs_lower;
            destination.attributes_per_core=source.attributes_per_core;
            destination.data_division=source.data_division;
            break;
            case 3:
            destination.summation_temp_thershold=source.fp_change_value;
            destination.rhs_upper=source.rhs_upper;
            destination.rhs_lower=source.rhs_lower;
            destination.attributes_per_core=source.attributes_per_core;
            destination.data_division=source.data_division;
            break;
            case 4:
            destination.rhs_upper=source.rhs_upper;
            destination.rhs_lower=source.rhs_lower;
            destination.attributes_per_core=source.attributes_per_core;
            destination.data_division=source.data_division;
            break;
            case 5:
            destination.rhs_lower=source.rhs_lower;
            destination.attributes_per_core=source.attributes_per_core;
            destination.data_division=source.data_division;
            break;
            case 6:
            destination.attributes_per_core=source.attributes_per_core;
            destination.data_division=source.data_division;
            break;
            case 7:
            destination.data_division=source.data_division;
            break;
        }
    }
}

vector<chromosome> genetic_algorithm::crossover(vector<chromosome>& population)
{
    vector<chromosome> new_gen;
    for(int a=0;a<population.size();a+=2)
    {
        if(a!=population.size()-1)
        {
            int crossover_index=get_random_number(0,7);//0 to crossover_index. If crossover_index = 7 than [0,1,2,3,4,5,6,7] && [8]
            chromosome new_chromosome1,new_chromosome2;
            //tranfer chromosome a data
            new_chromosome1.id=current_chromosome_id;
            current_chromosome_id++;
            chromosome_data_transfer(crossover_index,true,population[a],new_chromosome1);
            chromosome_data_transfer(crossover_index,false,population[a+1],new_chromosome1);
            new_gen.push_back(new_chromosome1);
            
            new_chromosome2.id=current_chromosome_id;
            current_chromosome_id++;
            chromosome_data_transfer(crossover_index,false,population[a],new_chromosome2);
            chromosome_data_transfer(crossover_index,true,population[a+1],new_chromosome2);
            new_gen.push_back(new_chromosome2);
        }
        else
        {   new_gen.push_back(population[a]);}
    }
    return new_gen;
}

vector<chromosome> genetic_algorithm::tournament_selection(vector<chromosome> population)
{
    vector<chromosome> selected_population;
    while(population.size()!=0)
    {
        int index1=get_random_number(0,population.size()-1);
        point1:
        int index2=get_random_number(0,population.size()-1);
        if(index1==index2)
        {   goto point1;}
        if(population[index1].fitness>population[index2].fitness)
        {
            selected_population.push_back(population[index1]);
            if(index1>index2)
            {
                population.erase(population.begin()+index1);
                population.erase(population.begin()+index2);
            }
            else
            {
                population.erase(population.begin()+index2);
                population.erase(population.begin()+index1);
            }
        }
        else
        {
            selected_population.push_back(population[index2]);
            if(index1>index2)
            {
                population.erase(population.begin()+index1);
                population.erase(population.begin()+index2);
            }
            else
            {
                population.erase(population.begin()+index2);
                population.erase(population.begin()+index1);
            }
        }
        if(population.size()==1)
        {   
            selected_population.push_back(population[0]);
            population.pop_back();
        }
    }
    return selected_population;
}

void genetic_algorithm::generate_initial_population()
{
    for(int a=0;a<population_size;a++)
    {
        chromosome c1;
        c1.id=a;
        c1.flatening_fx_enabled=get_random_bool();
        c1.extreame_weight_remover=get_random_bool();
        c1.zero_weight_remover=get_random_bool();
        c1.fp_change_value=get_random_number(fp_change_value_min,fp_change_value_max);
        c1.summation_temp_thershold=get_random_number(summation_temp_thershold_min,summation_temp_thershold_max);
        c1.rhs_upper=get_random_number(rhs_upper_min,rhs_upper_max);
        c1.rhs_lower=get_random_number(rhs_lower_min,rhs_lower_max);
        c1.attributes_per_core=get_random_number(attributes_per_core_min,attributes_per_core_max);
        c1.data_division=get_random_number(data_div_min,data_div_max);
        population.push_back(c1);
    }
    current_chromosome_id=population_size;
}

void genetic_algorithm::calc_fitness_threaded(int batch_size,vector<chromosome>& population)
{
    vector<segment_class*> segment_vec;
    vector<thread> thread_vec;
    for(int a=0;a<population.size();a+=batch_size)//this mechanism is to control amt of ram / vram used at a time
    {
        //prepare initial cdp and st
        thread_vec.resize(batch_size);
        for(int b=0;b<batch_size;b++)
        {
            if(a+b>=population.size())
            {   break;}
            segment_class *segment1=new segment_class(0,a+b,"default_segment");
            segment1->set_ds(ds);
            segment1->critical_variable=&population[a+b];
            segment1->set_critical_variable();
            //prepare_cdp_and_st_segment(segment1,*f_data_vector);
            thread_vec[b]=thread(prepare_cdp_and_st_segment,segment1,*f_data_vector);
            segment_vec.push_back(segment1);
        }
        for(int b=0;b<batch_size;b++)
        {   
            if(a+b>=population.size())
            {   break;}
            thread_vec[b].join();
        }
        thread_vec.clear();
        point1:
        vector<conflict_id> conflict_id_vec=simplex_solver();
        //int gh;cin>>gh;
        thread_vec.resize(batch_size);
        for(int b=0;b<conflict_id_vec.size();b+=batch_size)
        {
            for(int c=0;c<batch_size;c++)
            {
                if(b+c>=conflict_id_vec.size())
                {   break;}
                //thread_vec[c]=thread(handle_completed_table,&conflict_id_vec[b+c],b+c);
                handle_completed_table(&conflict_id_vec[b+c],b+c);
            }
            /*for(int c=0;c<batch_size;c++)
            {   
                if(b+c>=conflict_id_vec.size())
                {   break;}
                thread_vec[c].join();
            }*/
        }
        thread_vec.clear();
        cdp_vec.clear();
        st_vec.clear();
        cdp_vec.insert(cdp_vec.end(),cdp_vec_temp.begin(),cdp_vec_temp.end());
        st_vec.insert(st_vec.end(),st_vec_temp.begin(),st_vec_temp.end());
        cdp_vec_temp.clear();
        st_vec_temp.clear();
        //cout<<"\nnew st_vec size: "<<st_vec.size()<<" new cdp size: "<<cdp_vec.size();
        if(cdp_vec.size()!=0)
        {   goto point1;}
        //run test function
        //cout<<"\n\nwork done!!";
        /*for(int b=0;b<segment_vec.size();b++)
        {
            cout<<"\nsegment: "<<b;
            segment_vec[b]->print_all_path();
        }*/
        //cout<<"\nf_data_old: "<<f_data_vector->at(0).data.size()<<" f_data_new: "<<segment_vec[0]->f_data_vector[0].data.size();
        thread_vec.resize(segment_vec.size());
        for(int b=0;b<segment_vec.size();b++)
        {
            thread_vec[b]=thread(testing_for_each_label,&population[b+a],segment_vec[b],false);
            //testing_for_each_label(&population[b+a],segment_vec[b]);
        }
        for(int b=0;b<segment_vec.size();b++)
        {
            thread_vec[b].join();
            segment_vec[b]->clear_segment();
            delete segment_vec[b];
            //cout<<"\nb: "<<b<<" accuracy:"<<population[a+b].fitness;
        }
        cdp_vec.clear();
        segment_vec.clear();
        thread_vec.clear();
    }
}

vector<chromosome> genetic_algorithm::load_ga_state()
{
    vector<chromosome> population;
    string line,chunk;
    int counter=0,horizontal_index=0;
    ifstream file1("ga_state.csv",ios::out);
    while(file1)
    {
        file1>>line;
        if(strcasestr(line.c_str(),",")==0)
        {   continue;}
        if(counter>0)
        {
            chromosome cv;
            horizontal_index=0;
            for(int b=0;b<line.length();b++)
            {
                if(line.at(b)==',')
                {
                    switch(horizontal_index)
                    {
                        case 1:
                        if(stoi(chunk)==0)
                        {   cv.flatening_fx_enabled=false;}
                        else
                        {   cv.flatening_fx_enabled=true;}
                        break;
                        case 2:
                        if(stoi(chunk)==0)
                        {   cv.zero_weight_remover=false;}
                        else
                        {   cv.zero_weight_remover=true;}
                        break;
                        case 3:
                        if(stoi(chunk)==0)
                        {   cv.extreame_weight_remover=false;}
                        else
                        {   cv.extreame_weight_remover=true;}
                        break;
                        case 4:
                        cv.fp_change_value=stoi(chunk);
                        break;
                        case 5:
                        cv.summation_temp_thershold=stoi(chunk);
                        break;
                        case 6:
                        cv.rhs_upper=stoi(chunk);
                        break;
                        case 7:
                        cv.rhs_lower=stoi(chunk);
                        break;
                        case 8:
                        cv.attributes_per_core=stoi(chunk);
                        break;
                        case 9:
                        cv.data_division=stof(chunk);
                        break;
                        case 10:
                        cv.fitness=stof(chunk);
                        break;
                    }
                    chunk="";
                    horizontal_index++;
                }
                else
                {   chunk+=line.at(b);}
            }
            cv.fitness=stof(chunk);
            population.push_back(cv);
        }
        counter++;
        line="";
    }
    file1.close();
    return population;
}

void genetic_algorithm::save_ga_state()
{
    ofstream file1("ga_state.csv",ios::out);
    file1<<"sino,flatening_fx_enabled,zero_weight_remover,extreame_weight_remover,fp_change_value,summation_temp_thershold,rhs_upper,rhs_lower,attributes_per_core,data_division,fitness\n";
    for(int a=0;a<population.size();a++)
    {
        file1<<a<<",";
        if(population[a].flatening_fx_enabled)
        {   file1<<1<<",";}
        else
        {   file1<<0<<",";}
        if(population[a].zero_weight_remover)
        {   file1<<1<<",";}
        else
        {   file1<<0<<",";}
        if(population[a].extreame_weight_remover)
        {   file1<<1<<",";}
        else
        {   file1<<0<<",";}
        file1<<population[a].fp_change_value<<",";
        file1<<population[a].summation_temp_thershold<<",";
        file1<<population[a].rhs_upper<<",";
        file1<<population[a].rhs_lower<<",";
        file1<<population[a].attributes_per_core<<",";
        file1<<population[a].data_division<<",";
        file1<<population[a].fitness<<"\n";
    }
    file1.close();
}

chromosome genetic_algorithm::start_genetic_algorithm(int batch_size)
{
    no_of_genes_to_mutate=8*mutation_percentage/100;
    if(no_of_genes_to_mutate==0)
    {   no_of_genes_to_mutate=3;}
    else if(no_of_genes_to_mutate>8)//to handle more than 100 percent condition, which is illegal.
    {   no_of_genes_to_mutate=3;}
    population=load_ga_state();
    if(population.size()==0)
    {   generate_initial_population();}
    else
    {
        population_size=population.size();
        cout<<"\nPrevious ga state file found, so overiding population size....";
        cout<<"\nNew population size: "<<population_size;
        //print_population(population);
    }
    //cout<<"\ndone!!";
    //int gh;cin>>gh;
    //fitness calculation
    cout<<"\n\nCalculation initial population fitness...";
    calc_fitness_threaded(batch_size,population);
    int total_value=0;
    for(int a=0;a<ga_iterations;a++)
    {
        //selection
        vector<chromosome> selected_parents=tournament_selection(population);
        //crossover
        vector<chromosome> new_gen=crossover(selected_parents);
        selected_parents.clear();
        //mutation
        mutation(new_gen);
        //fitness calculation
        calc_fitness_threaded(batch_size,new_gen);
        //adding new gen to population pool
        population.insert(population.end(),new_gen.begin(),new_gen.end());
        //sorting population pool according to fitness
        sort(population.begin(),population.end(),comparator);
        //removing the last 50 chromosome with lowest fitness
        population.erase(population.end()-(population.size()-population_size),population.end());
        total_value=0;
        for(int b=0;b<population.size();b++)
        {   
            //cout<<"\nfitness= "<<population[b].fitness;
            total_value+=population[b].fitness;
        }
        if((a+1)!=0 && (a+1)%25==0)
        {   save_ga_state();}
        //save_chromosome(population[0]);
        cout<<"\nIteration: "<<a<<", total_population_fitness: "<<total_value<<", max_fitness: "<<population[0].fitness;
    }
    //int gh;cin>>gh;
    return population[0];
}

void genetic_algorithm::print_population(vector<chromosome>& population)
{
    for(int a=0;a<population.size();a++)
    {
        cout<<"\n"<<population[a].flatening_fx_enabled<<","<<population[a].zero_weight_remover<<","<<population[a].extreame_weight_remover<<","<<population[a].fp_change_value<<","<<population[a].summation_temp_thershold<<","<<population[a].rhs_upper<<","<<population[a].rhs_lower<<","<<population[a].attributes_per_core<<","<<population[a].data_division<<","<<population[a].fitness;
    }
}

void genetic_algorithm::save_chromosome(chromosome& chromosome)
{
    ofstream file1("critical_var.txt",ios::app);
    file1<<"\nflatening_fx_enabled: "<<chromosome.flatening_fx_enabled;
    file1<<"\nzero_weight_remover: "<<chromosome.zero_weight_remover;
    file1<<"\nextreame_weight_remover: "<<chromosome.extreame_weight_remover;
    file1<<"\nfp_change_value: "<<chromosome.fp_change_value;
    file1<<"\nsummation_temp_threshold: "<<chromosome.summation_temp_thershold;
    file1<<"\nrhs_upper: "<<chromosome.rhs_upper;
    file1<<"\nrhs_lower: "<<chromosome.rhs_lower;
    file1<<"\nattributes_per_core: "<<chromosome.attributes_per_core;
    file1<<"\ndata_division: "<<chromosome.data_division;
    file1<<"\nfitness: "<<chromosome.fitness;
    file1.close();
}

genetic_algorithm::genetic_algorithm(unsigned int &iterations,unsigned int &pop_size,unsigned int &mutation_percentage1,int &data_div_max1)
{
    ga_iterations=iterations;
    population_size=pop_size;
    mutation_percentage=mutation_percentage1;
    data_div_max=data_div_max1;
}