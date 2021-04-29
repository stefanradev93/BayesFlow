Search.setIndex({docnames:["bayesflow","index","modules","support","tutorial_notebooks/Meta_Workflow","tutorial_notebooks/Model_Comparison_Workflow","tutorial_notebooks/Parameter_Estimation_Workflow","tutorials"],envversion:{"sphinx.domains.c":2,"sphinx.domains.changeset":1,"sphinx.domains.citation":1,"sphinx.domains.cpp":3,"sphinx.domains.index":1,"sphinx.domains.javascript":2,"sphinx.domains.math":2,"sphinx.domains.python":2,"sphinx.domains.rst":2,"sphinx.domains.std":2,"sphinx.ext.viewcode":1,nbsphinx:3,sphinx:56},filenames:["bayesflow.rst","index.rst","modules.rst","support.rst","tutorial_notebooks/Meta_Workflow.ipynb","tutorial_notebooks/Model_Comparison_Workflow.ipynb","tutorial_notebooks/Parameter_Estimation_Workflow.ipynb","tutorials.rst"],objects:{"":{bayesflow:[0,0,0,"-"]},"bayesflow.amortizers":{MetaAmortizer:[0,1,1,""],MultiModelAmortizer:[0,1,1,""],SingleModelAmortizer:[0,1,1,""]},"bayesflow.amortizers.MetaAmortizer":{__init__:[0,2,1,""],call:[0,2,1,""],compare_models:[0,2,1,""],sample_from_model:[0,2,1,""]},"bayesflow.amortizers.MultiModelAmortizer":{__init__:[0,2,1,""],call:[0,2,1,""],sample:[0,2,1,""]},"bayesflow.amortizers.SingleModelAmortizer":{__init__:[0,2,1,""],call:[0,2,1,""],sample:[0,2,1,""]},"bayesflow.buffer":{MemoryReplayBuffer:[0,1,1,""]},"bayesflow.buffer.MemoryReplayBuffer":{__init__:[0,2,1,""],_buffer:[0,3,1,""],capacity:[0,3,1,""],sample:[0,2,1,""],size_in_batches:[0,3,1,""],store:[0,2,1,""]},"bayesflow.default_settings":{MetaDictSetting:[0,1,1,""],Setting:[0,1,1,""]},"bayesflow.default_settings.MetaDictSetting":{__init__:[0,2,1,""]},"bayesflow.default_settings.Setting":{__init__:[0,2,1,""]},"bayesflow.diagnostics":{plot_calibration_curves:[0,4,1,""],plot_confusion_matrix:[0,4,1,""],plot_expected_calibration_error:[0,4,1,""],plot_sbc:[0,4,1,""],true_vs_estimated:[0,4,1,""]},"bayesflow.exceptions":{ConfigurationError:[0,5,1,""],LossError:[0,5,1,""],OperationNotSupportedError:[0,5,1,""],SimulationError:[0,5,1,""],SummaryStatsError:[0,5,1,""]},"bayesflow.helpers":{build_meta_dict:[0,4,1,""],clip_gradients:[0,4,1,""],merge_left_into_right:[0,4,1,""]},"bayesflow.losses":{heteroscedastic_loss:[0,4,1,""],kl_dirichlet:[0,4,1,""],kl_latent_space:[0,4,1,""],log_loss:[0,4,1,""],maximum_mean_discrepancy:[0,4,1,""],mean_squared_loss:[0,4,1,""],meta_amortized_loss:[0,4,1,""]},"bayesflow.models":{GenerativeModel:[0,1,1,""],MetaGenerativeModel:[0,1,1,""],SimpleGenerativeModel:[0,1,1,""]},"bayesflow.models.MetaGenerativeModel":{__call__:[0,2,1,""],__init__:[0,2,1,""],generative_models:[0,3,1,""],model_prior:[0,3,1,""],n_models:[0,3,1,""],param_padding:[0,3,1,""]},"bayesflow.models.SimpleGenerativeModel":{__call__:[0,2,1,""],__init__:[0,2,1,""],data_transform:[0,3,1,""],param_transform:[0,3,1,""],prior:[0,3,1,""],simulator:[0,3,1,""]},"bayesflow.networks":{ConditionalCouplingLayer:[0,1,1,""],CouplingNet:[0,1,1,""],EquivariantModule:[0,1,1,""],EvidentialNetwork:[0,1,1,""],HeteroscedasticRegressionNetwork:[0,1,1,""],InvariantModule:[0,1,1,""],InvariantNetwork:[0,1,1,""],InvertibleNetwork:[0,1,1,""],Permutation:[0,1,1,""],RegressionNetwork:[0,1,1,""],SequenceNet:[0,1,1,""]},"bayesflow.networks.ConditionalCouplingLayer":{__init__:[0,2,1,""],call:[0,2,1,""]},"bayesflow.networks.CouplingNet":{__init__:[0,2,1,""],call:[0,2,1,""]},"bayesflow.networks.EquivariantModule":{__init__:[0,2,1,""],call:[0,2,1,""]},"bayesflow.networks.EvidentialNetwork":{__init__:[0,2,1,""],call:[0,2,1,""],evidence:[0,2,1,""],predict:[0,2,1,""],sample:[0,2,1,""]},"bayesflow.networks.HeteroscedasticRegressionNetwork":{__init__:[0,2,1,""],call:[0,2,1,""]},"bayesflow.networks.InvariantModule":{__init__:[0,2,1,""],call:[0,2,1,""]},"bayesflow.networks.InvariantNetwork":{__init__:[0,2,1,""],call:[0,2,1,""]},"bayesflow.networks.InvertibleNetwork":{__init__:[0,2,1,""],call:[0,2,1,""],forward:[0,2,1,""],inverse:[0,2,1,""],sample:[0,2,1,""]},"bayesflow.networks.Permutation":{__init__:[0,2,1,""],call:[0,2,1,""]},"bayesflow.networks.RegressionNetwork":{__init__:[0,2,1,""],call:[0,2,1,""]},"bayesflow.networks.SequenceNet":{__init__:[0,2,1,""],call:[0,2,1,""]},"bayesflow.trainers":{BaseTrainer:[0,1,1,""],MetaTrainer:[0,1,1,""],ModelComparisonTrainer:[0,1,1,""],ParameterEstimationTrainer:[0,1,1,""]},"bayesflow.trainers.BaseTrainer":{__init__:[0,2,1,""],load_pretrained_network:[0,2,1,""],simulate_and_train_offline:[0,2,1,""],train_offline:[0,2,1,""],train_online:[0,2,1,""],train_rounds:[0,2,1,""]},"bayesflow.trainers.MetaTrainer":{__init__:[0,2,1,""]},"bayesflow.trainers.ModelComparisonTrainer":{__init__:[0,2,1,""],train_offline:[0,2,1,""]},"bayesflow.trainers.ParameterEstimationTrainer":{__init__:[0,2,1,""],train_experience_replay:[0,2,1,""]},bayesflow:{amortizers:[0,0,0,"-"],buffer:[0,0,0,"-"],default_settings:[0,0,0,"-"],diagnostics:[0,0,0,"-"],exceptions:[0,0,0,"-"],helpers:[0,0,0,"-"],losses:[0,0,0,"-"],models:[0,0,0,"-"],networks:[0,0,0,"-"],trainers:[0,0,0,"-"]}},objnames:{"0":["py","module","Python module"],"1":["py","class","Python class"],"2":["py","method","Python method"],"3":["py","attribute","Python attribute"],"4":["py","function","Python function"],"5":["py","exception","Python exception"]},objtypes:{"0":"py:module","1":"py:class","2":"py:method","3":"py:attribute","4":"py:function","5":"py:exception"},terms:{"0001":4,"0005":0,"00300":1,"005":[5,6],"032":5,"03899":1,"06082":1,"06281":1,"100":[0,5,6],"1000":[0,1,4,6],"106":5,"10629":1,"107":5,"11s":5,"12765":[4,6],"128":[4,5],"133":4,"134":4,"135":4,"136":4,"137":4,"1391":4,"1392":4,"1393":4,"1394":4,"1395":4,"13s":5,"14min":5,"150":5,"1538":4,"1539":4,"1540":4,"1541":4,"1542":4,"157":4,"158":4,"159":4,"15min":5,"15s":5,"160":4,"161":4,"162":4,"163":4,"164":4,"165":4,"1901":1,"1904":[4,6],"199":4,"1990":4,"1991":4,"1992":4,"1993":4,"1994":4,"1e4":6,"200":[1,4,5,6],"20000":4,"2003":1,"2004":1,"2005":1,"201":4,"2010":1,"2018":0,"202":4,"2020":1,"203":4,"23s":5,"250":[4,6],"284":4,"285":4,"286":4,"287":4,"288":4,"2par":5,"300":[0,4,5,6],"311":4,"312":4,"313":4,"314":4,"315":4,"328":4,"329":4,"330":4,"3par":5,"400":[5,6],"468":4,"469":4,"470":4,"471":4,"472":4,"4min":5,"4par":5,"500":[4,5,6],"5000":[1,4,6],"563":4,"564":4,"565":4,"566":4,"567":4,"600":6,"623":4,"624":4,"625":4,"626":4,"627":4,"661":4,"662":4,"663":4,"664":4,"665":4,"670":4,"671":4,"672":4,"673":4,"674":4,"6e2":5,"800":6,"878":4,"879":4,"880":4,"881":4,"882":4,"884":4,"885":4,"886":4,"887":4,"888":4,"8f2727":[4,6],"95179":4,"abstract":0,"b\u00fcrkner":1,"boolean":0,"class":[0,4,6],"default":[0,1,6],"final":1,"float":[0,5],"function":[0,1],"import":[0,4,5,6],"int":[0,4,5,6],"k\u00f6the":1,"new":[5,6],"return":[0,1,4,5,6],"short":[1,7],"super":4,"true":[0,4,6],"try":[0,4],"var":[4,6],"while":6,For:1,Ihe:0,One:[0,4],The:[0,1,6,7],Then:0,Use:1,Uses:5,Using:1,__call__:[0,4],__iadd__:4,__init__:[0,4],_apply_gradi:4,_as_graph_el:4,_buffer:0,_cached_valu:4,_core:4,_ctx:4,_decayed_lr:4,_dense_var_to_tensor:4,_dtype:4,_forward_infer:0,_handl:4,_hyper:4,_kwarg:0,_maybe_set_handle_data:4,_notokstatusexcept:4,_prepar:4,_prepare_loc:4,_read_variable_op:4,_result:4,_train_step:4,_transform_unaggregated_gradi:4,_true_param:5,a_c:6,abc:0,ablin:0,about:1,abov:1,abs:[1,4,5,6],absolut:0,abspath:[4,5,6],accept:1,accepted_result_typ:4,access:4,accord:4,account:1,across:0,activ:[4,5,6],actual:0,adam:[0,4],add:[5,6],add_n:4,addit:0,advoc:0,after:0,alessandro:1,algorithm:1,align:1,all:[0,4],allow:0,along:0,alpha:[0,4,6],alpha_h:5,alpha_m:5,alpha_n:5,alreadi:[0,6],amort:[1,2],anaconda3:4,analyt:[1,4,6],ani:1,annot:0,append:[4,5,6],appli:0,apply_gradi:4,apply_st:4,approach:1,approxim:[0,1,4],arang:[4,5],architectur:[0,1,4],ardizzon:1,arg:[0,4,5],arg_nam:0,argmax:4,argument:[0,1,4,5,6],aris:1,arrai:[0,4,5,6],array_op:4,arxiv:[1,4,6],as_ref:4,assert:5,asset:[4,6],associ:1,assum:[0,4,5],astyp:[4,5,6],atan:4,attach:4,attempt:[0,1,4],attribut:0,autoreload:[4,5,6],avail:1,axarr:[4,6],axi:[0,4,6],ba12d72aea3b:4,back:4,backprop:[0,4,6],backpropag:0,backward:[0,4],bar:4,base:[0,1,4],basetrain:0,batch:[0,1,4,5,6],batch_simul:[4,6],batch_siz:[0,1,4,5,6],bay:0,bayes_flow:4,bayesflow:[5,7],bayesian:[0,1],bayesianmultivariatet:4,been:0,behavior:1,below:1,benjamin:1,beta:1,beta_h:5,beta_m:5,beta_n:5,betanalpha:[4,6],between:[0,1],bf_meta:[4,6],bin:0,black:[4,6],block:[0,4],bloem:1,blow:4,blue:0,bool:[0,4],bootstrap:1,both:[0,4],bound:[0,4,6],buffer:2,build:4,build_meta_dict:0,cal_err:0,cal_prob:0,calcul:0,calibr:[0,4,6],call:[0,1,4,6],callabl:[0,4],can:[0,1],capabl:1,capac:[0,6],care:6,carlo:0,case_studi:[4,6],cast:4,centralstoragestrategi:4,chain:[0,4],chang:1,check:[0,1],checkpoint:0,checkpoint_path:[0,4],choic:[4,5],cinn:[0,4],clamp:4,clip:0,clip_gradi:[0,4],clip_method:[0,4],clip_valu:[0,4],cm2:5,cmap:0,code:1,cognit:[1,4,6],colocate_with:4,color:[0,4,6],colormap:0,combin:0,compare_model:0,comparison:[0,7],compil:5,complex:1,composit:1,comput:[0,1,4,6],concat:4,concaten:[0,4,6],cond_arr:6,condit:[0,4,6],conditionalcouplinglay:[0,4],confer:1,configur:0,configurationerror:0,confus:0,connect:[0,1,6],consid:0,consider:1,consist:0,constant:5,contain:[0,5],content:2,context:4,continu:0,contract:[4,6],control:0,conv:0,conversion_func:4,convert:[4,5,6],convert_to_eager_tensor:4,convert_to_tensor:4,cornerston:1,correct:4,correspond:[0,1,4],could:6,coupl:[0,4],couplingnet:0,covid:1,cpu:[5,6],creat:[0,4,6],ctx:4,current:[0,1,4,5],curv:0,custom:0,data:[0,1,4,5],data_dim:0,data_transform:0,dataset:[0,4],ddm:6,deep:[0,1],def:[4,5,6],default_dict:0,default_set:2,defin:[0,1],demand:0,demo:5,dens:[0,4],dense_arg:5,dense_h1_arg:4,dense_h2_arg:4,dense_s1_arg:6,dense_s2_arg:6,dense_s3_arg:6,densiti:0,depend:1,depict:1,deriv:0,describ:1,design:1,desktop:4,despin:[4,6],det:0,detail:1,determin:[0,4],devic:4,diag:4,diagnost:[2,4,5,6],dict:[0,5],dictionari:[0,4],differ:[0,1],differenti:1,diffus:6,diffusion_2_cond:6,diffusion_condit:6,diffusion_model_dataset:6,diffusion_tri:6,dimens:[0,1],dimension:0,dir:0,direct:[0,4],dirichlet:0,disabl:4,discrep:0,diseas:1,dispatch:4,distribut:[0,1],distro:0,diverg:0,dm_batch_simul:0,dm_prior:0,doc:1,doe:0,don:3,dot:0,dpi:0,draw:[1,4,5,6],dtype:[4,6],dtype_hint:4,durat:5,dynam:1,e_k:5,e_leak:5,e_na:5,each:[0,1,4,5,6],earli:1,easi:1,efun:5,either:[0,4],element:1,els:[4,5,6],elu:[4,6],emploi:1,empti:6,encod:[0,4],end:[0,1],engin:0,ensur:1,entir:6,entri:0,enumer:[4,6],env:4,ep_num:0,epoch:[0,1,4,5,6],equal:[0,4,5],equat:[1,6],equiv_dim:0,equivari:0,equivariantmodul:0,error:0,estim:[0,7],evid:0,evidence_net:0,evidenti:[0,1],evidential_meta:5,evidential_net:5,evidentialnetwork:[0,5],evolut:1,exampl:[0,1],example_object:0,excel:1,except:[2,4],exchang:1,executing_eagerli:4,exist:0,exp:[4,5],expand_dim:5,expected_calibration_error:5,experi:[0,1],experiment:4,experimental_aggregate_gradi:4,extend:0,extens:6,extern:1,eye:4,factori:0,fail:4,fall:4,fals:[0,4],fast:[0,1,5],featur:3,field:0,figsiz:[0,4,6],figur:[0,1],filenam:0,first:1,fix:[0,5,6],flag:[0,4],flat:[4,6],float32:[0,4,6],focu:1,folder:0,follow:[0,1,7],font:0,font_siz:0,fontsiz:[4,6],form:1,format:4,forward:[0,4,5,6],forward_model1:[0,5],forward_model2:[0,5],forward_model3:5,found:1,framework:4,free:1,from:[0,1,4,5,6],full:0,func:4,functool:5,futur:0,g_j:5,g_l:5,gaussian:[0,1],gbar_k:5,gbar_l:5,gbar_m:5,gbar_na:5,gen_array_op:4,gen_resource_variable_op:4,gener:[0,1,4],generate_data:4,generate_multiple_dataset:4,generative_model:[0,1,4,5,6],generativemodel:[0,1,4,5,6],germani:1,get:4,github:[3,4,6],given:[0,1,4],global:0,global_norm:0,glorot_uniform:[4,5,6],gpu:4,gradient:[0,4],grads_and_var:4,graph:4,grid:[4,6],h_inf:5,handl:[0,4],happier:4,have:[0,3,4],helper:2,here:[0,1,5,6],hesit:3,heteroscedast:0,heteroscedastic_loss:0,heteroscedasticregressionnetwork:0,high:[5,6],higher:1,histogram:0,hold:[0,4],hot:[0,4,5],html:[4,6],http:[1,4,6],hypothet:1,i_dur:5,i_input:5,idea:1,ident:4,ieee:1,ignore_exist:4,iid:1,illustr:1,implement:[0,1,4,5],impli:[0,1],inactiv:5,inch:0,incorpor:1,increment:6,index:[0,1,4],indic:[0,4],individu:1,induc:1,inf:0,infecti:1,infer:[0,1,4,6],inference_net:[0,1,6],inform:1,initi:[0,4,5,6],inn:[0,4],inp_dim:[0,4],input:[0,4,5],input_dim:0,inquiri:3,instanc:[0,1,4,6],int32:[4,5],integ:4,integr:[0,1],interest:[0,4],interfac:0,intern:[0,1],interv:0,intract:1,intrins:0,introduct:7,inv:[0,4],invari:[0,1,6],invariantbayesflow:4,invariantcouplingnet:4,invariantmodul:0,invariantnetwork:[0,1,6],invers:[0,4],invert:[0,1,4,6],invertiblenetwork:[0,1,6],ipynb:1,ipython:4,issu:3,iter:[0,1,4],iterations_per_epoch:[0,1,5,6],jacobian:[0,4],join:[4,5,6],jointli:1,just:5,keep:0,kei:[0,5],kera:[0,4,5],kernel:0,kernel_initi:[4,5],keyboardinterrupt:4,keyword:0,kinet:5,kl_dirichlet:0,kl_latent_spac:[0,4],kullback:0,kwarg:[0,4],label:6,lambd:0,last:[0,4],latent:[0,1],layer:[0,1,4,5],layout:0,ldot:0,learn:[0,1],learnabl:0,learning_r:[0,4],left_dict:0,leibler:0,level:1,lib:4,librari:1,likelihood:[0,1],linearsegmentedcolormap:0,list:[0,4],load:[0,6],load_ext:[4,5,6],load_pretrained_network:0,loc:4,local_step:4,log:[0,4],log_det_j:[0,4],log_loss:0,logarithm:0,logloss:0,loss:[1,2,4,5,6],loss_valu:0,losserror:0,low:[5,6],lower:[0,4,6],lr_t:4,lstm:[0,5],m_idx:[0,4,5],m_indic:[0,4],m_indices_v:4,m_inf:5,m_oh:4,m_pred:0,m_prob:0,m_rep:4,m_true:[0,4,5],m_var:0,mail:3,make:[0,4],manag:0,mandatori:0,mandatory_field:0,manner:1,map:1,math:4,math_op:4,mathcal:5,mathrm:0,matplotlib:[0,4,5,6],matrix:[0,4],matter:0,max:[0,4],max_it:6,max_step:6,max_to_keep:0,maximum:0,maximum_mean_discrep:0,mean:[0,4,6],mean_squared_loss:0,meet:0,memori:[0,1,4],memoryreplaybuff:0,merg:0,merge_left_into_right:0,merged_dict:0,merten:1,meta5:4,meta:[0,7],meta_amort:4,meta_amortized_loss:0,meta_dict:0,meta_generative_model:0,metaamort:0,metadictset:0,metagenerativemodel:0,metatrain:[0,4],method:[0,1],metric:4,minimum:0,mmd:0,model1_params_prior:[0,5],model2_params_prior:[0,5],model3_params_prior:5,model:[2,7],model_comparison_workflow:1,model_indic:[0,4,5],model_indices_b:4,model_nam:0,model_prior:[0,4,5],modelcomparisontrain:[0,5],modul:[1,2,4],mont:0,more:1,most:[1,4],mu_sampl:4,mu_scal:4,mua:5,multi:0,multimodelamort:[0,5],multipl:[0,4,6],multivari:1,multivariate_norm:4,multivariate_t:4,multivariatet:4,must:0,n_batch:[0,4],n_bin:0,n_coupling_lay:[4,6],n_dataset:0,n_dens:5,n_dense_h1:4,n_dense_h2:4,n_dense_s1:6,n_dense_s2:6,n_dense_s3:6,n_equiv:6,n_inf:5,n_max:6,n_min:6,n_model:[0,4,5],n_ob:[0,1,4,5,6],n_out1:4,n_out2:4,n_out:[0,4],n_out_dim:0,n_param:[0,1,4,6],n_post_samples_sbc:[4,6],n_sampl:[0,1,4,6],n_samples_posterior:[4,6],n_sbc:[4,6],n_sim:[0,4,5,6],n_sim_:[4,6],n_step:6,n_trial:6,n_trials_c1:6,n_trials_c2:6,n_val:4,name:[0,4],ndarrai:[4,5,6],ndt:[4,6],necessari:0,need:[0,1],nest:0,net:[0,4],network:[1,2,4,5],neural:[0,1,4],newaxi:4,next:1,njit:[4,5,6],nois_fact:5,non:[1,5],none:[0,4,5],norm:0,normal:[0,4,6],note:[0,4],notebook:[1,5,7],nothign:[4,5],notimpl:4,now:1,num_featur:0,numba:[4,5,6],number:[0,4,5,6],numpi:[4,5,6],object:0,obs_data:[0,1],observ:[0,1,6],obtain:[0,1,4],occur:0,offlin:[0,1],one:[0,4,5],ones:6,onli:0,onlin:[0,1],open:3,oper:[0,4],operationnotsupportederror:0,ops:4,optim:[0,1,4],optimizer_v2:4,option:[0,1],order:[0,1],org:[1,4,6],origin:0,other:[0,1],otherwis:4,our:1,out:[0,1,4],out_activ:5,out_dim:0,out_evid:0,out_infer:0,outbreak:1,output:[0,1,4,5,6],over:[0,1,6],overwrit:0,p_inf:5,p_sampl:[4,6],p_val:[4,5],packag:[1,2,4,7],pad:0,page:1,pair:[0,4],panda:[4,5,6],paper:1,par:5,parallel:[0,4],param:[0,4,5,6],param_dim:0,param_mean:[4,6],param_nam:[0,4,6],param_pad:0,param_sampl:[4,6],param_transform:0,paramet:[0,4,7],parameter_estimation_workflow:1,parameterestimationtrain:[0,1,6],parameterserverstrategi:4,params_b:4,params_m:4,params_m_v:4,params_rep:4,params_sbc:[4,6],partial:5,pass:[0,4],path:[0,4,5,6],per:[0,1],perform:[0,1,4],permut:[0,1,4,6],phase:1,pleas:[0,1,3,5],plot:[0,4,6],plot_calibration_curv:[0,5],plot_confusion_matrix:[0,5],plot_expected_calibration_error:0,plot_sbc:[0,4,6],plt:[0,4,5,6],pm_sampl:0,point:5,pool:0,posit:0,possibl:1,post_cont:[4,6],post_mean:[4,6],post_sampl:0,post_std:[4,6],post_var:[4,6],post_z_scor:[4,6],posterior:[0,1],potenti:[0,1],pre:[0,4],pred_mean:0,pred_var:0,predict:0,predicted_param:0,preferred_dtyp:4,preprint:1,present:1,previou:[5,6],prime:1,principl:[4,6],principled_bayesian_workflow:[4,6],print:0,prior:[0,1],prior_a:[4,6],prior_b:[4,6],prior_mu:4,prior_n:6,prior_sampl:6,prior_scal:4,prior_var:[4,6],prob:0,probabilist:1,probabl:0,problem:0,process:[0,1,6],prod:4,profil:4,progress:4,project:4,propag:4,protect:4,provid:[0,5,7],pylint:4,pyplot:[0,4,5,6],python:[0,4],pywrap_tf:4,r2_score:4,radev:1,rais:[0,4],randint:6,randn:5,random:[1,4,5,6],randomli:0,rang:[4,5,6],rate:0,read:1,read_and_set_handl:4,read_valu:4,read_variable_op:4,readi:1,readvariableop:4,real:0,recent:4,recommend:1,reddi:1,reduce_mean:4,reduce_sum:4,regress:0,regressionnetwork:0,regular:0,reload:6,reload_ext:6,relu:[4,5,6],rep:4,rep_params_m:4,replai:[0,1],repres:[0,4],represent:1,request:3,requir:1,resolv:4,resourc:4,resource_variable_op:4,respons:1,result:[0,4],ret:4,revers:[0,4],review:1,right_dict:0,risk:0,round:[0,1],rt_c1:6,rt_c2:6,rts:6,run:[0,1],rvs:4,s_arg:[4,6],sampl:[0,1,4,5,6],sample_from_model:0,save:0,scalar:0,scale_sampl:4,scale_scal:4,scan:0,scatter:[0,4,6],scipi:4,score:[4,6],scratch:4,sde:1,seaborn:[4,5,6],search:1,second:0,see:1,seed:4,self:[0,4],sequenc:0,sequencenet:[0,5],sequenti:[0,4,5],seri:1,set:[0,1],set_titl:[4,6],set_xlabel:[4,6],set_xlim:[4,6],set_ylabel:[4,6],set_ylim:[4,6],shall:0,shape:[0,1,4,5,6],should:1,show:0,shown:0,signatur:0,sim:0,sim_data:[0,4,5,6],sim_data_b:4,sim_data_v:4,sim_per_round:[0,5,6],simpl:0,simple_generative_model:0,simplegenerativemodel:0,simpli:1,simul:[0,1],simulate_and_train_offlin:[0,6],simulate_data:4,simulationerror:0,singl:[0,1,4,6],singlemodelamort:[0,1,6],site:4,size:[0,4,5,6],size_in_batch:0,sklearn:4,slow:5,sns:[4,5,6],softplu:5,solut:1,some:[1,4],somewher:6,sourc:[0,1],source_sampl:0,space:1,specifi:[0,4,5],speed:5,split:[0,4,6],sqrt:6,squar:0,stack:[4,6],starter:1,stat:4,state:5,statist:[0,1],std:[4,6],steadi:5,step:[0,4,6],stochast:[1,5],store:[0,4],str:0,string:0,structur:0,stuff:6,submodul:2,subplot:[4,6],sum:4,sum_dim:0,sum_meta:6,sum_stat:0,summar:[0,4],summari:[0,1,6],summary_dim:[0,1,4],summary_net:[0,1,5,6],summary_stat:0,summarystatserror:0,superclass:0,support:[0,1],sure:[0,4],symmetri:1,synthet:0,sys:[4,5,6],system:1,t_arg:[4,6],t_off:5,t_on:5,t_post:5,tackl:1,take:[0,6],talt:0,tape:4,target:4,target_sampl:0,tau_h:5,tau_m:5,tau_max:5,tau_n:5,tau_p:5,tau_v_inv:5,teh:1,tensor:[0,4],tensorflow:[0,4,5,6],tensorflowdev:4,term:0,test:[0,4],tfe_py_fastpathexecut:4,than:1,thereof:0,theta:[0,4,5,6],theta_dim:[0,4,5,6],theta_est:0,theta_j:5,theta_sampl:[0,4],theta_test:0,theta_tru:0,thi:[0,5],those:1,though:[0,4],three:[0,1],through:[0,4,6],thu:1,tight_layout:[4,6],time:[0,1,4,5,6],to_categor:4,to_numpi:[0,4],todo:0,too:1,total:[0,4,5,6],toward:[4,6],tprior:4,trace:4,trace_kwarg:4,trace_nam:4,traceback:4,train:[0,1],train_experience_replai:[0,6],train_offlin:[0,4,5,6],train_onlin:[0,1,5,6],train_round:[0,5,6],trainable_vari:4,trainer:[1,2,4,5,6],trajectori:1,transact:1,transform:[0,4],treat:0,trial:6,true_model_indic:0,true_param:[0,4,6],true_vs_estim:[0,4,6],tstep:5,tun:4,tupl:0,tutori:1,tutorial_notebook:1,two:0,type:[0,1],typeerror:4,typic:1,uncertainti:0,underli:0,uniform:[0,4,5,6],union:0,unit:[4,5,6],unknown:0,unspecifi:0,unused_oth:4,updat:4,upfront:6,upper:[4,6],use:[0,6],used:[0,1],user:[0,5,6],user_dict:0,using:0,usual:[0,1],util:4,v_c:6,v_inf:5,valid:[0,1,4,6],valu:[0,4,5],valueerror:[0,4],var_devic:4,var_dtyp:4,var_list:4,variabl:[0,4,6],varianc:0,variou:1,vector:0,version:[0,1,4],via:[0,1,4,6],violat:0,voltag:5,voss:1,walkthrough:1,wall:[5,6],want:0,weight:0,well:0,when:[0,1,4],where:[0,4],wherein:[0,4],whether:[0,4],which:[0,1,6],whole:1,whose:1,whye:1,within:1,without:0,word:1,work:[0,1],workflow:[1,7],wrap:[0,4],wrapper:4,write:3,x_dim:[0,4],x_params_m:4,x_sbc:[4,6],yee:1,you:[0,1,3],your:[0,5],z_dim:4,z_normal_sampl:4,zero:[0,4,6],zeros_lik:5,zip:[4,6]},titles:["bayesflow package","Welcome the documentation of BayesFlow!","bayesflow","Support","Meta Workflow","Model Comparison Workflow","Parameter Estimation Workflow","Tutorials"],titleterms:{"100":4,Using:6,adequaci:[4,6],amort:[0,4,5,6],base:[5,6],bayesflow:[0,1,2,4,6],bayesian:[4,6],buffer:0,calibr:5,check:[4,5,6],comparison:[1,5],compuat:[4,6],conceptu:1,content:0,custom:6,data:6,default_set:0,diagnost:0,dirti:[4,6],document:1,estim:[1,4,5,6],exampl:[4,5,6],except:0,experi:6,eyechart:[4,6],faith:[4,6],helper:0,indic:1,intern:6,joint:1,loss:0,memoryless:1,meta:4,model:[0,1,4,5,6],modul:0,network:[0,6],offlin:[4,5,6],onlin:[5,6],overview:1,packag:0,paramet:[1,5,6],perform:5,postdict:[4,6],posterior:[4,6],pre:6,predict:[4,5,6],prior:[4,5,6],quick:[4,6],replai:6,round:[5,6],sensit:[4,6],set:[4,5,6],simul:[4,5,6],state:1,stateless:1,submodul:0,support:3,tabl:1,train:[4,5,6],trainer:0,tutori:7,welcom:1,workflow:[4,5,6]}})