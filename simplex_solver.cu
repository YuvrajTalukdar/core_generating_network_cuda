#include"core_class.h"
#include<thrust/device_vector.h>
#include<thrust/host_vector.h>

/*Test functions*/

void check(cudaError x) {
    fprintf(stderr, "%s\n", cudaGetErrorString(x));
}
void copy_table_to_ram(simplex_table_cuda *st_d);

/*Simplex solver functions and kernel*/
int shared_memory_size=0;
__global__ void pivot_row_modifier(simplex_table_cuda *st_arr,float *pe_arr,int *p_row_arr,int *p_col_arr,char *completion_code)//ok check
{
    if(completion_code[blockIdx.x]=='0')
    {
        switch(threadIdx.x)
        {
            case 0:
            st_arr[blockIdx.x].r_id[p_row_arr[blockIdx.x]].basic=st_arr[blockIdx.x].c_id[p_col_arr[blockIdx.x]].basic;
            break;
            case 1:
            st_arr[blockIdx.x].r_id[p_row_arr[blockIdx.x]].id=st_arr[blockIdx.x].c_id[p_col_arr[blockIdx.x]].id;
            break;
            case 2:
            st_arr[blockIdx.x].r_id[p_row_arr[blockIdx.x]].rhs=st_arr[blockIdx.x].c_id[p_col_arr[blockIdx.x]].rhs;;
            break;
            case 3:
            st_arr[blockIdx.x].r_id[p_row_arr[blockIdx.x]].slack=st_arr[blockIdx.x].c_id[p_col_arr[blockIdx.x]].slack;
            break;
            case 4:
            st_arr[blockIdx.x].r_id[p_row_arr[blockIdx.x]].theta=st_arr[blockIdx.x].c_id[p_col_arr[blockIdx.x]].theta;
            break;
            default:
        }

        if(threadIdx.x<st_arr[blockIdx.x].basic_var_size_col)
        {   st_arr[blockIdx.x].basic_var[p_row_arr[blockIdx.x]*st_arr[blockIdx.x].basic_var_size_col+threadIdx.x]/=pe_arr[blockIdx.x];}
        else if(threadIdx.x>=st_arr[blockIdx.x].basic_var_size_col && threadIdx.x<(st_arr[blockIdx.x].basic_var_size_col+st_arr[blockIdx.x].slack_var_size_col))
        {
            int slack_col_index=threadIdx.x-st_arr[blockIdx.x].basic_var_size_col;
            st_arr[blockIdx.x].slack_var[p_row_arr[blockIdx.x]*st_arr[blockIdx.x].slack_var_size_col+slack_col_index]/=pe_arr[blockIdx.x];
        }
        else if(threadIdx.x==(st_arr[blockIdx.x].basic_var_size_col+st_arr[blockIdx.x].slack_var_size_col))
        {   st_arr[blockIdx.x].rhs[p_row_arr[blockIdx.x]]/=pe_arr[blockIdx.x];}
    }
}

__global__ void rest_of_row_modifier(simplex_table_cuda *st_arr,int *p_row_arr,int *p_col_arr,/*float *multiplying_element_matrix,*/int largest_row,char *completion_code)//ok check
{
    //row is blockIdx.y
    if(completion_code[blockIdx.x]=='0')
    {
        if(threadIdx.x<st_arr[blockIdx.x].basic_var_size_row)
        {
            if(threadIdx.x!=p_row_arr[blockIdx.x])//all row accept pivot row
            {
                if(blockIdx.y!=p_col_arr[blockIdx.x])
                {
                    float multiplying_element;
                    if(p_col_arr[blockIdx.x]<st_arr[blockIdx.x].basic_var_size_col)
                    {   multiplying_element=st_arr[blockIdx.x].basic_var[threadIdx.x*st_arr[blockIdx.x].basic_var_size_col+p_col_arr[blockIdx.x]];}
                    else
                    {   
                        int index=p_col_arr[blockIdx.x]-st_arr[blockIdx.x].basic_var_size_col;
                        multiplying_element=st_arr[blockIdx.x].slack_var[threadIdx.x*st_arr[blockIdx.x].slack_var_size_col+index];
                    }
                    if(blockIdx.y<(st_arr[blockIdx.x].basic_var_size_col+st_arr[blockIdx.x].slack_var_size_col))
                    {
                        if(blockIdx.y<st_arr[blockIdx.x].basic_var_size_col)//basic_point
                        {   
                            st_arr[blockIdx.x].basic_var[threadIdx.x*st_arr[blockIdx.x].basic_var_size_col+blockIdx.y]-=(multiplying_element*st_arr[blockIdx.x].basic_var[p_row_arr[blockIdx.x]*st_arr[blockIdx.x].basic_var_size_col+blockIdx.y]);
                        }
                        else if(blockIdx.y>=st_arr[blockIdx.x].basic_var_size_col && blockIdx.y<(st_arr[blockIdx.x].basic_var_size_col+st_arr[blockIdx.x].slack_var_size_col))//slack_point
                        {
                            int slack_col_index=blockIdx.y-st_arr[blockIdx.x].basic_var_size_col;
                            st_arr[blockIdx.x].slack_var[threadIdx.x*st_arr[blockIdx.x].slack_var_size_col+slack_col_index]-=(multiplying_element*st_arr[blockIdx.x].slack_var[p_row_arr[blockIdx.x]*st_arr[blockIdx.x].slack_var_size_col+slack_col_index]);
                        }
                    }
                    else if(blockIdx.y==st_arr[blockIdx.x].basic_var_size_col+st_arr[blockIdx.x].slack_var_size_col)//rhs
                    {
                        st_arr[blockIdx.x].rhs[threadIdx.x]-=multiplying_element*st_arr[blockIdx.x].rhs[p_row_arr[blockIdx.x]];   
                    }
                }
            }
        }
    }
}

__global__ void p_col_modifier(simplex_table_cuda *st_arr,int *p_row_arr,int *p_col_arr,char *completion_code)
{
    if(completion_code[blockIdx.x]=='0')
    {
        if(threadIdx.x<st_arr[blockIdx.x].basic_var_size_row && threadIdx.x!=p_row_arr[blockIdx.x])
        {
            if(p_col_arr[blockIdx.x]<st_arr[blockIdx.x].basic_var_size_col)
            {
                st_arr[blockIdx.x].basic_var[threadIdx.x*st_arr[blockIdx.x].basic_var_size_col+p_col_arr[blockIdx.x]]=0;
            }
            else
            {
                int index=p_col_arr[blockIdx.x]-st_arr[blockIdx.x].basic_var_size_col;
                st_arr[blockIdx.x].slack_var[threadIdx.x*st_arr[blockIdx.x].slack_var_size_col+index]=0;
            }
        }
    }
}

void simplex_table_modifier(simplex_table_cuda *st_d_arr,int* row_with_negative_slack_d,float *pe_d_arr,int *p_row_arr_d,int *p_col_arr_d,char *completion_code_d,int largest_col,int largest_row,cudaStream_t *stream1)//ok check
{
    //copy_table_to_ram(st_d_arr);//for testing
    //display_st(st_vec[11]);//for testing
    //int *row_with_negative_slack_test=(int*)malloc(sizeof(int)*st_vec.size());
    //int *p_row_test=(int*)malloc(sizeof(int)*st_vec.size());//for testing
    //int *p_col_test=(int*)malloc(sizeof(int)*st_vec.size());//for testing
    //float *pe_test=(float*)malloc(sizeof(float)*st_vec.size());//for testing
    //cudaMemcpy(pe_test,pe_d_arr,sizeof(float)*st_vec.size(),cudaMemcpyDeviceToHost);
    //cudaMemcpy(p_col_test,p_col_arr_d,sizeof(int)*st_vec.size(),cudaMemcpyDeviceToHost);//for testing
    //cudaMemcpy(p_row_test,p_row_arr_d,sizeof(int)*st_vec.size(),cudaMemcpyDeviceToHost);//for testing
    //cudaMemcpy(row_with_negative_slack_test,row_with_negative_slack_d,sizeof(int)*st_vec.size(),cudaMemcpyDeviceToHost);//for testing
    //cout<<"\nrow_with_negative_slack:"<<row_with_negative_slack_test[11];//for testing
    //cout<<"\np_row:"<<p_row_test[11]<<" p_col:"<<p_col_test[11]<<" pe:"<<pe_test[11]<<" ";//for testing
    //int gh;cin>>gh;

    largest_col++;//extr one for rhs
    //pivot row modifier
    if(largest_col<5)
    {   largest_col=5;}
    pivot_row_modifier<<<st_vec.size(),largest_col,shared_memory_size,*stream1>>>(st_d_arr,pe_d_arr,p_row_arr_d,p_col_arr_d,completion_code_d);
    //rest of the row modifiew

    dim3 block_vec(st_vec.size(),largest_col,1);
    rest_of_row_modifier<<<block_vec,largest_row,shared_memory_size,*stream1>>>(st_d_arr,p_row_arr_d,p_col_arr_d,/*multiplying_element_matrix,*/largest_row,completion_code_d);
    p_col_modifier<<<st_vec.size(),largest_row,shared_memory_size,*stream1>>>(st_d_arr,p_row_arr_d,p_col_arr_d,completion_code_d);
    cudaStreamSynchronize(*stream1);

    //copy_table_to_ram(st_d_arr);//for testing
    //display_st(st_vec[0]);//for testing
    //char *completion_code_test=(char*)malloc(sizeof(char)*st_vec.size());
    //cudaMemcpy(completion_code_test,completion_code_d,sizeof(char)*st_vec.size(),cudaMemcpyDeviceToHost);
    //if(completion_code_test[0]=='1')
    //{   cout<<"\n\nst solved!!!!!!";}
}

__global__ void termination_condition_checker_kernel(simplex_table_cuda *st_arr,char *completion_code,int no_of_tables)//ok check
{
    int index=blockIdx.x*32+threadIdx.x;
    if(index<no_of_tables)
    {
        if(completion_code[index]=='0')
        {
            bool status=true;
            for(int a=0;a<st_arr[index].basic_var_size_row;a++)
            {
                if(st_arr[index].r_id[a].slack)
                {   
                    if(st_arr[index].slack_var[a*st_arr[index].slack_var_size_col+st_arr[index].r_id[a].id-st_arr[index].basic_var_size_col]<0)
                    {   
                        if(st_arr[index].rhs[a]>=0)
                        {   status=false;break;}
                    }
                }
            }
            if(status)
            {   completion_code[index]='1';}
        }
    }
}

bool termination_condition_checker(simplex_table_cuda *st_d_arr,int largest_row,char *completion_code,char *completion_code_d,cudaStream_t *stream1,int no_of_blocks)//ok check
{
    int termination_count=0;
    termination_condition_checker_kernel<<<no_of_blocks,32,shared_memory_size,*stream1>>>(st_d_arr,completion_code_d,st_vec.size());//block,thread
    cudaStreamSynchronize(*stream1);
    cudaMemcpy(completion_code,completion_code_d,sizeof(char)*st_vec.size(),cudaMemcpyDeviceToHost);
    for(int a=0;a<st_vec.size();a++)
    {
        if(completion_code[a]!='0')
        {   termination_count++;}
    }
    if(termination_count==st_vec.size())
    {   return true;}
    else
    {   return false;}
}

__global__ void find_row_with_negative_slack_and_p_col_kernel(simplex_table_cuda *st_arr,int *row_with_negative_slack,int *p_col_arr,char *completion_code,int no_of_tables)//ok check
{
    int index=blockIdx.x*32+threadIdx.x;
    if(index<no_of_tables)
    {
        if(completion_code[index]=='0')
        {
            row_with_negative_slack[index]=-1;
            for(int a=0;a<st_arr[index].basic_var_size_row;a++)
            {
                if(st_arr[index].r_id[a].slack)
                {
                    if(st_arr[index].slack_var[a*st_arr[index].slack_var_size_col+(st_arr[index].r_id[a].id-st_arr[index].basic_var_size_col)]<0 && st_arr[index].rhs[a]>=0)//originally it was just rhs>0, but now i feel it shouls be >=. Need further testing
                    {   row_with_negative_slack[index]=a;break;}
                }
            }
            if(row_with_negative_slack[index]==-1)
            {   completion_code[index]='5';}
            else
            {
                int col=-1;
                for(int a=0;a<st_arr[index].basic_var_size_col;a++)
                {
                    if(st_arr[index].basic_var[row_with_negative_slack[index]*st_arr[index].basic_var_size_col+a]>0)
                    {   col=a;break;}
                }
                if(col==-1)
                {
                    for(int a=0;a<st_arr[index].slack_var_size_col;a++)
                    {   
                        if(st_arr[index].slack_var[row_with_negative_slack[index]*st_arr[index].slack_var_size_col+a]>0)
                        {   col=a+st_arr[index].basic_var_size_col;break;}
                    }
                }
                if(col==-1)
                {   completion_code[index]='2';}
                p_col_arr[index]=col;
            }
        }
    }
}

void conflicting_data_finder(int st_table_index,vector<conflict_id> &conflict_id_vec)//ok check
{
    for(int a=0;a<st_vec[st_table_index]->r_id_size;a++)
    {
        if(st_vec[st_table_index]->r_id[a].slack && st_vec[st_table_index]->slack_var[a*st_vec[st_table_index]->slack_var_size_col+st_vec[st_table_index]->r_id[a].id-st_vec[st_table_index]->basic_var_size_col]<0 && st_vec[st_table_index]->rhs[a]>0)
        {   conflict_id_vec[st_table_index].id_vec.push_back(st_vec[st_table_index]->r_id[a].id-st_vec[st_table_index]->basic_var_size_col);}
    }
}

__global__ void calc_theta_kernel(simplex_table_cuda *st_arr,int *pivote_col_index,char *completion_code)//ok check
{
    if(completion_code[blockIdx.x]=='0' && threadIdx.x<st_arr[blockIdx.x].basic_var_size_row)
    {
        if(pivote_col_index[blockIdx.x]<st_arr[blockIdx.x].basic_var_size_col)
        {
            if(st_arr[blockIdx.x].basic_var[threadIdx.x*st_arr[blockIdx.x].basic_var_size_col+pivote_col_index[blockIdx.x]]==0)
            {   st_arr[blockIdx.x].theta[threadIdx.x]=0;}
            else
            {   st_arr[blockIdx.x].theta[threadIdx.x]=st_arr[blockIdx.x].rhs[threadIdx.x]/(double)st_arr[blockIdx.x].basic_var[threadIdx.x*st_arr[blockIdx.x].basic_var_size_col+pivote_col_index[blockIdx.x]];}
        }
        else
        {
            int temp_col_index=pivote_col_index[blockIdx.x]-st_arr[blockIdx.x].basic_var_size_col;
            if(st_arr[blockIdx.x].slack_var[threadIdx.x*st_arr[blockIdx.x].slack_var_size_col+temp_col_index]==0)
            {   st_arr[blockIdx.x].theta[threadIdx.x]=0;}
            else
            {   st_arr[blockIdx.x].theta[threadIdx.x]=st_arr[blockIdx.x].rhs[threadIdx.x]/(double)st_arr[blockIdx.x].slack_var[threadIdx.x*st_arr[blockIdx.x].slack_var_size_col+temp_col_index];}
        }
    }
}

__global__ void get_pivot_row_element_kernel(simplex_table_cuda *st_arr,int *p_row_arr,int *p_col_arr,float *pe_arr,char* completion_code,int no_of_tables)//ok check
{
    int index=blockIdx.x*32+threadIdx.x;
    if(index<no_of_tables)
    {
        if(completion_code[index]=='0')
        {
            p_row_arr[index]=-1;
            double smallest_value=-1;
            for(int a=0;a<st_arr[index].basic_var_size_row;a++)
            {
                if(st_arr[index].theta[a]>0)
                {
                    if(smallest_value==-1 || smallest_value>st_arr[index].theta[a])
                    {
                        smallest_value=st_arr[index].theta[a];
                        p_row_arr[index]=a;
                    }
                }
            }
            if(p_row_arr[index]<0)
            {   completion_code[index]='3';}
            else//get pe
            {
                if(p_col_arr[index]<st_arr[index].basic_var_size_col)
                {
                    pe_arr[index]=st_arr[index].basic_var[p_row_arr[index]*st_arr[index].basic_var_size_col+p_col_arr[index]];
                }
                else
                {
                    int slack_p_col=p_col_arr[index]-st_arr[index].basic_var_size_col;
                    pe_arr[index]=st_arr[index].slack_var[p_row_arr[index]*st_arr[index].slack_var_size_col+slack_p_col];
                }
            }
        }
    }
}

void check_for_cyclic_bug(int *p_col_arr_d,int *p_row_arr_d,vector<buffer> &buffer_obj_vec,simplex_table_cuda *st_d_arr,char *completion_code,char *completion_code_d)//need to be checked. The algorithm used here is new and much better as it should have exceptionally low false positives.
{
    int size_large=4,size_small=2;//large must be disible by small
    int *p_col_arr,*p_row_arr;
    p_col_arr=(int*)malloc(sizeof(int)*st_vec.size());
    cudaMemcpy(p_col_arr,p_col_arr_d,sizeof(int)*st_vec.size(),cudaMemcpyDeviceToHost);
    p_row_arr=(int*)malloc(sizeof(int)*st_vec.size());
    cudaMemcpy(p_row_arr,p_row_arr_d,sizeof(int)*st_vec.size(),cudaMemcpyDeviceToHost);
    cudaMemcpy(completion_code,completion_code_d,sizeof(char)*st_vec.size(),cudaMemcpyDeviceToHost);
    bool cc_changed=false;
    for(int a=0;a<st_vec.size();a++)
    {
        if(completion_code[a]=='0')
        {
            if(buffer_obj_vec[a].p_col_index_small.size()<size_small)
            {
                buffer_obj_vec[a].p_col_index_small.push_back(p_col_arr[a]);
                buffer_obj_vec[a].p_row_index_small.push_back(p_row_arr[a]);
            }
            else
            {
                buffer_obj_vec[a].p_col_index_small.push_back(p_col_arr[a]);
                buffer_obj_vec[a].p_row_index_small.push_back(p_row_arr[a]);
                buffer_obj_vec[a].p_col_index_small.erase(buffer_obj_vec[a].p_col_index_small.begin());
                buffer_obj_vec[a].p_row_index_small.erase(buffer_obj_vec[a].p_row_index_small.begin());   
            }
            if(buffer_obj_vec[a].p_col_index.size()<size_large)
            {
                buffer_obj_vec[a].p_col_index.push_back(p_col_arr[a]);
                buffer_obj_vec[a].p_row_index.push_back(p_row_arr[a]);
            }
            else
            {
                for(int b=0;b<buffer_obj_vec[a].p_row_index.size()-size_small+1;b++)
                {
                    int match=0;
                    for(int c=0;c<buffer_obj_vec[a].p_col_index_small.size();c++)
                    {
                        if(buffer_obj_vec[a].p_row_index[b+c]==buffer_obj_vec[a].p_row_index_small[c] && 
                           buffer_obj_vec[a].p_col_index[b+c]==buffer_obj_vec[a].p_col_index_small[c])
                        {   match++;}
                    }
                    if(match==size_small)
                    {   
                        completion_code[a]='4';
                        cc_changed=true;
                        break;
                    }
                }
                if(completion_code[a]!='4')
                {
                    buffer_obj_vec[a].p_col_index.push_back(p_col_arr[a]);
                    buffer_obj_vec[a].p_row_index.push_back(p_row_arr[a]);
                    buffer_obj_vec[a].p_col_index.erase(buffer_obj_vec[a].p_col_index.begin());
                    buffer_obj_vec[a].p_row_index.erase(buffer_obj_vec[a].p_row_index.begin());
                }
            }
        }
    }
    if(cc_changed)
    {   cudaMemcpy(completion_code_d,completion_code,sizeof(char)*st_vec.size(),cudaMemcpyHostToDevice);}
    free(p_row_arr);
    free(p_col_arr);
}

void free_simplex_table_from_vram(simplex_table_cuda *st_d_arr)//ok check
{
    simplex_table_cuda *ram_arr=(simplex_table_cuda*)malloc(sizeof(simplex_table_cuda)*st_vec.size());
    cudaMemcpy(ram_arr,st_d_arr,sizeof(simplex_table_cuda)*st_vec.size(),cudaMemcpyDeviceToHost);
    for(int a=0;a<st_vec.size();a++)
    {
        cudaFree(ram_arr[a].basic_var);
        cudaFree(ram_arr[a].c_id);
        cudaFree(ram_arr[a].r_id);
        cudaFree(ram_arr[a].rhs);
        cudaFree(ram_arr[a].slack_var);
        cudaFree(ram_arr[a].theta);
    }
    cudaFree(st_d_arr);
    free(ram_arr);
}

vector<conflict_id> pivot_element_finder(simplex_table_cuda *st_d_arr)
{
    int largest_row_size=0,largest_col_size=0;
    for(int a=0;a<st_vec.size();a++)
    {
        int col_size=st_vec[a]->basic_var_size_col+st_vec[a]->slack_var_size_col;
        if(largest_col_size<col_size)
        {   largest_col_size=col_size;}
        int row_size=st_vec[a]->basic_var_size_row;
        if(largest_row_size<row_size)
        {   largest_row_size=row_size;}
    }
    char *completion_code=(char*)malloc(sizeof(char)*st_vec.size());
    for(int a=0;a<st_vec.size();a++)
    {   completion_code[a]='0';}//0=not complete, 1=complete, 2=conflict_found, 3=bad_p_row, 4=cyclic_bug, 5=row_with_negative_element not found
    char *completion_code_d;
    cudaMalloc(&completion_code_d,sizeof(char)*st_vec.size());
    cudaMemcpy(completion_code_d,completion_code,sizeof(char)*st_vec.size(),cudaMemcpyHostToDevice);

    vector<conflict_id> conflict_id_vec(st_vec.size());
    int *row_with_negative_slack_d;
    int *p_col_arr_d,*p_row_arr_d;
    cudaMalloc(&p_col_arr_d,sizeof(int)*st_vec.size());
    cudaMalloc(&p_row_arr_d,sizeof(int)*st_vec.size());
    vector<buffer> buffer_obj_vec(st_vec.size());
    cudaStream_t stream1;
    cudaStreamCreate(&stream1);
    float *pe_d_arr;
    //int iteration=0;
    cudaMalloc(&pe_d_arr,sizeof(float)*st_vec.size());
    cudaMalloc(&row_with_negative_slack_d,sizeof(float)*st_vec.size());
    int no_of_blocks=st_vec.size()/32,no_of_threads_rem=st_vec.size()%32;
    if(no_of_threads_rem!=0)
    {   no_of_blocks++;}
    do
    {
        //cout<<"\niteration: "<<iteration;
        find_row_with_negative_slack_and_p_col_kernel<<<no_of_blocks,32,shared_memory_size,stream1>>>(st_d_arr,row_with_negative_slack_d,p_col_arr_d,completion_code_d,st_vec.size());
        calc_theta_kernel<<<st_vec.size(),largest_row_size,shared_memory_size,stream1>>>(st_d_arr,p_col_arr_d,completion_code_d);
        get_pivot_row_element_kernel<<<no_of_blocks,32,shared_memory_size,stream1>>>(st_d_arr,p_row_arr_d,p_col_arr_d,pe_d_arr,completion_code_d,st_vec.size());
        cudaStreamSynchronize(stream1);
        check_for_cyclic_bug(p_col_arr_d,p_row_arr_d,buffer_obj_vec,st_d_arr,completion_code,completion_code_d);
        simplex_table_modifier(st_d_arr,row_with_negative_slack_d,pe_d_arr,p_row_arr_d,p_col_arr_d,completion_code_d,largest_col_size,largest_row_size,&stream1);
        //iteration++;
    } 
    while(!termination_condition_checker(st_d_arr,largest_row_size,completion_code,completion_code_d,&stream1,no_of_blocks));
    cudaStreamDestroy(stream1);
    buffer_obj_vec.clear();
    cudaMemcpy(completion_code,completion_code_d,sizeof(char)*st_vec.size(),cudaMemcpyDeviceToHost);
    int complete=0,conflict=0;
    copy_table_to_ram(st_d_arr);
    for(int a=0;a<st_vec.size();a++)
    {
        conflict_id_vec[a].completion_code=completion_code[a];
        if(completion_code[a]=='1')
        {   complete++;}
        else if(completion_code[a]=='0')//engine shutdown before completing work.
        {   cout<<"\nERROR! engine shutdown before completing work. a: "<<a<<" completion_code: "<<completion_code[a];}
        else
        {   conflicting_data_finder(a,conflict_id_vec);conflict++;
            if(conflict_id_vec.at(a).id_vec.size()==0)
            {   
                cout<<"\nFailed to read conflict ids! "<<conflict_id_vec.at(a).id_vec.size();
                cout<<"\ncc: "<<completion_code[a];
                //display_st(st_vec[a]);
                //int *row_with_negative_slack=(int*)malloc(sizeof(int)*st_vec.size());
                //cudaMemcpy(row_with_negative_slack,row_with_negative_slack_d,sizeof(int)*st_vec.size(),cudaMemcpyDeviceToHost);
                //int *p_col_arr=(int*)malloc(sizeof(int)*st_vec.size());
                //cudaMemcpy(p_col_arr,p_col_arr_d,sizeof(int)*st_vec.size(),cudaMemcpyDeviceToHost);
                //int *p_row_arr=(int*)malloc(sizeof(int)*st_vec.size());
                //cudaMemcpy(p_row_arr,p_row_arr_d,sizeof(int)*st_vec.size(),cudaMemcpyDeviceToHost);
                //cout<<"\np_col: "<<p_col_arr[a]<<" p_row: "<<p_row_arr[a]<<" row_with-ve_slack: "<<row_with_negative_slack[a];
                //int gh;cin>>gh;
            }
        }
    }
    //cout<<"\ncomplete: "<<complete<<" conflict: "<<conflict;
    free(completion_code);
    cudaFree(completion_code_d);
    cudaFree(p_col_arr_d);
    cudaFree(p_row_arr_d);
    cudaFree(pe_d_arr);
    cudaFree(row_with_negative_slack_d);
    
    return conflict_id_vec;
}

simplex_table_cuda* copy_table_to_vram(simplex_table_cuda *st_d_arr)//ok check
{
    for(int a=0;a<st_vec.size();a++)
    {
        st_d_arr[a].basic_var_size_col=st_vec[a]->basic_var_size_col;
        st_d_arr[a].basic_var_size_row=st_vec[a]->basic_var_size_row;
        cudaMalloc(&st_d_arr[a].basic_var,sizeof(float)*st_d_arr[a].basic_var_size_col*st_d_arr[a].basic_var_size_row);
        cudaMemcpy(st_d_arr[a].basic_var,st_vec[a]->basic_var,sizeof(float)*st_d_arr[a].basic_var_size_col*st_d_arr[a].basic_var_size_row,cudaMemcpyHostToDevice);
        //free(st_vec[a]->basic_var);

        st_d_arr[a].c_id_size=st_vec[a]->c_id_size;
        cudaMalloc(&st_d_arr[a].c_id,sizeof(id)*st_d_arr[a].c_id_size);
        cudaMemcpy(st_d_arr[a].c_id,st_vec[a]->c_id,sizeof(id)*st_d_arr[a].c_id_size,cudaMemcpyHostToDevice);
        //free(st_vec[a]->c_id);

        st_d_arr[a].r_id_size=st_vec[a]->r_id_size;
        cudaMalloc(&st_d_arr[a].r_id,sizeof(id)*st_d_arr[a].r_id_size);
        cudaMemcpy(st_d_arr[a].r_id,st_vec[a]->r_id,sizeof(id)*st_d_arr[a].r_id_size,cudaMemcpyHostToDevice);
        //free(st_vec[a]->r_id);

        st_d_arr[a].slack_var_size_col=st_vec[a]->slack_var_size_col;
        st_d_arr[a].slack_var_size_row=st_vec[a]->slack_var_size_row;
        cudaMalloc(&st_d_arr[a].slack_var,sizeof(float)*st_d_arr[a].slack_var_size_col*st_d_arr[a].slack_var_size_row);
        cudaMemcpy(st_d_arr[a].slack_var,st_vec[a]->slack_var,sizeof(float)*st_d_arr[a].slack_var_size_col*st_d_arr[a].slack_var_size_row,cudaMemcpyHostToDevice);
        //free(st_vec[a]->slack_var);

        st_d_arr[a].rhs_size=st_vec[a]->rhs_size;
        cudaMalloc(&st_d_arr[a].rhs,sizeof(double)*st_d_arr[a].rhs_size);
        cudaMemcpy(st_d_arr[a].rhs,st_vec[a]->rhs,sizeof(double)*st_d_arr[a].rhs_size,cudaMemcpyHostToDevice);
        //free(st_vec[a]->rhs);
        
        cudaMalloc(&st_d_arr[a].theta,sizeof(double)*st_d_arr[a].basic_var_size_row);
    }
    simplex_table_cuda *device_arr;
    cudaMalloc(&device_arr,sizeof(simplex_table_cuda)*st_vec.size());
    cudaMemcpy(device_arr,st_d_arr,sizeof(simplex_table_cuda)*st_vec.size(),cudaMemcpyHostToDevice);
    free(st_d_arr);
    return device_arr;
}

void copy_table_to_ram(simplex_table_cuda *st_d_arr)//ok check
{
    simplex_table_cuda *ram_arr=(simplex_table_cuda*)malloc(sizeof(simplex_table_cuda)*st_vec.size());
    cudaMemcpy(ram_arr,st_d_arr,sizeof(simplex_table_cuda)*st_vec.size(),cudaMemcpyDeviceToHost);
    for(int a=0;a<st_vec.size();a++)
    {
        cudaMemcpy(st_vec[a]->basic_var,ram_arr[a].basic_var,sizeof(float)*ram_arr[a].basic_var_size_col*ram_arr[a].basic_var_size_row,cudaMemcpyDeviceToHost);
        cudaMemcpy(st_vec[a]->c_id,ram_arr[a].c_id,sizeof(id)*ram_arr[a].c_id_size,cudaMemcpyDeviceToHost);
        cudaMemcpy(st_vec[a]->r_id,ram_arr[a].r_id,sizeof(id)*ram_arr[a].r_id_size,cudaMemcpyDeviceToHost);
        cudaMemcpy(st_vec[a]->slack_var,ram_arr[a].slack_var,sizeof(float)*ram_arr[a].slack_var_size_col*ram_arr[a].slack_var_size_row,cudaMemcpyDeviceToHost);
        cudaMemcpy(st_vec[a]->rhs,ram_arr[a].rhs,sizeof(double)*ram_arr[a].rhs_size,cudaMemcpyDeviceToHost);
    }
    free(ram_arr);
}

vector<conflict_id> simplex_solver()
{
    //auto start = high_resolution_clock::now();
    simplex_table_cuda *st_d_arr;
    st_d_arr=(simplex_table_cuda*)malloc(sizeof(simplex_table_cuda)*st_vec.size());
    st_d_arr=copy_table_to_vram(st_d_arr);
    vector<conflict_id> conflict_id_vec=pivot_element_finder(st_d_arr);
    free_simplex_table_from_vram(st_d_arr);
    int complete=0,conflict=0;
    for(int a=0;a<conflict_id_vec.size();a++)
    {
        if(conflict_id_vec[a].id_vec.size()==0)
        {   complete++;}
        else
        {   conflict++;}
    }
    //cout<<"\ncomplete2: "<<complete<<" conflict2: "<<conflict;
    //auto end = high_resolution_clock::now();
    //auto duration = duration_cast<microseconds>(end - start); 
    //cout<<"\n\nduration= "<<duration.count()/pow(10,6)<<" sec";
    //int gh;cin>>gh;
    return conflict_id_vec;
}