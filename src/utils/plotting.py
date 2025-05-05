
import pickle as pkl
import numpy as np
import matplotlib.pyplot as plt
import yaml
import pandas as pd
import os
import src.utils.utils as utils

def get_TTE(solution_dict,num_nodes):
    TTE= []
    zero_solution = solution_dict[0][2]
    return [solution_dict[node][2]-zero_solution for node in num_nodes]

def get_liftup_graph(node_list,total_nodes,setting,do_greedy,do_GA,do_CELF,do_CFR,do_CFR_heuristic,do_random,do_greedy_simulated,do_full,T):

    #plt.style.use("science")
    plt.rcParams['text.usetex'] = True
    plt.rcParams['pgf.texsystem'] = 'pdflatex'
    plt.rcParams['pgf.preamble'] = r'\usepackage{amsmath}'
    plt.rcParams['font.family'] = 'serif'
    plt.rcParams['font.serif'] = ['Times'] # Change to the desired serif font
    plt.rcParams['axes.labelsize'] = 26 # Change to the desired label size
    plt.rcParams['axes.titlesize'] = 20 # Change to the desired title size
    plt.rcParams['xtick.labelsize'] = 22  # Adjust size of x-axis tick labels
    plt.rcParams['ytick.labelsize'] = 22  # Adjust size of y-axis tick labels
    #plt.rcParams["mathtext.default"] = "it"

    # Open the file in binary mode
    my_path = "data/allocations/" + setting + "/"
    if do_full:
        extra = "full_"
    else:
        extra = ""

    with open(my_path + extra + 'degree_solution_dict_test.pkl', 'rb') as file:
        degree_solution_dict = pkl.load(file)

    if do_GA:

        with open(my_path + extra +'ga_solution_dict_test.pkl', 'rb') as file:
            ga_solution_dict = pkl.load(file)
    # print(ga_solution_dict)

    with open(my_path + extra +'single_discount_solution_dict_test.pkl', 'rb') as file:
        # Load the data from the .pkl file
        single_discount_solution_dict = pkl.load(file)
    #print(single_discount_solution_dict)

    if do_greedy:
        with open(my_path + extra +'greedy_solution_dict_test.pkl', 'rb') as file:
            # Load the data from the .pkl file
            greedy_solution_dict = pkl.load(file)
        #print(greedy_solution_dict)
    with open(my_path + extra +'random_solution_dict_test.pkl', 'rb') as file: 
        # Load the data from the .pkl file
        random_solution_dict = pkl.load(file)

    if do_CFR:
        with open(my_path + extra +'CFR_solution_dict_test.pkl', 'rb') as file: 
            # Load the data from the .pkl file
            CFR_solution_dict = pkl.load(file)
    # #celf
    if do_CELF:
        with open(my_path + extra +'celf_solution_dict_test.pkl', 'rb') as file:
            # Load the data from the .pkl file
            celf_solution_dict = pkl.load(file)
    # # #greedy_simulated
    if do_greedy_simulated:
        with open(my_path + extra +'greedy_simulated_solution_dict_test.pkl', 'rb') as file:
            # Load the data from the .pkl file
            greedy_simulated_solution_dict = pkl.load(file)
    # #CFR_heuristic
    #     # Load the data from the .pkl file
    #     CFR_heuristic_solution_dict = pkl.load(file)
    if do_greedy:
        TTE_greedy =np.array(get_TTE(greedy_solution_dict,node_list))
        #print("TTE_greedy",TTE_greedy)
    if do_GA:
        TTE_ga = np.array(get_TTE(ga_solution_dict,node_list))
    TTE_degree = np.array(get_TTE(degree_solution_dict,node_list))
    TTE_single_discount = np.array(get_TTE(single_discount_solution_dict,node_list))
    TTE_random = np.array(get_TTE(random_solution_dict,node_list))
    if do_CFR:
        TTE_CFR = np.array(get_TTE(CFR_solution_dict,node_list))
    if do_CELF:
        TTE_celf = np.array(get_TTE(celf_solution_dict,node_list))
    if do_greedy_simulated:
        TTE_greedy_simulated = np.array(get_TTE(greedy_simulated_solution_dict,node_list))
    #TTE_CFR_heuristic = get_TTE(CFR_heuristic_solution_dict,node_list)
    node_list = np.array(node_list)*100
    legend_fig, legend_ax = plt.subplots()

# Create dummy lines for the legend
    legend_lines = []
    legend_labels = []

    if do_greedy:
        legend_lines.append(legend_ax.plot([], [], color='green', linestyle='-', marker='o', label='OTAPI-GR')[0])
        legend_labels.append('OTAPI-GR')
    if do_GA:
        legend_lines.append(legend_ax.plot([], [], color='red', linestyle='-', marker='*', label='OTAPI-GA')[0])
        legend_labels.append('OTAPI-GA')
    legend_lines.append(legend_ax.plot([], [], color='blue', linestyle='-', marker='s', label='Degree')[0])
    legend_labels.append('Degree')
    legend_lines.append(legend_ax.plot([], [], color='black', linestyle='-', marker='p', label='SD')[0])
    legend_labels.append('SD')
    if do_CFR:
        legend_lines.append(legend_ax.plot([], [], color='purple', linestyle='-', marker='v', label='CFR')[0])
        legend_labels.append('CFR')
    if do_CELF:
        legend_lines.append(legend_ax.plot([], [], color='pink', linestyle='-', marker='d', label='CELF IC')[0])
        legend_labels.append('CELF IC')
    if do_greedy_simulated:
        legend_lines.append(legend_ax.plot([], [], color='brown', linestyle='-', marker='P', label='Upper Bound')[0])
        legend_labels.append('Upper Bound')
    
    

    # Create the legend using dummy lines and labels
    my_path_results = "figures/" +  setting + "/"
    if not os.path.exists(my_path_results):
        os.makedirs(my_path_results)

    legend_ax.legend(handles=legend_lines, labels=legend_labels, loc='center', fontsize=20)
    legend_ax.axis('off')  # Hide axis for the legend figure

    # Save the legend to a separate PDF file
    legend_path = my_path_results + extra + 'Legend.pdf'
    legend_fig.savefig(legend_path, dpi=300, bbox_inches='tight', format='pdf')

    # Close the legend figure to free up resources
    plt.close()

    # Show the plot (optional)
    # plt.show()


    #i can be used to cut off part of the curve
    i =0
    #Find the index closest:
    
    node_list = node_list[i:]
    node_list = np.array(node_list)
    if do_greedy:
        TTE_greedy = TTE_greedy[i:]
    if do_GA:
        TTE_ga = TTE_ga[i:]
    TTE_degree = TTE_degree[i:]
    TTE_single_discount = TTE_single_discount[i:]
    TTE_random = TTE_random[i:]
    if do_CFR:
        TTE_CFR = TTE_CFR[i:]
    if do_CELF:
        TTE_celf = TTE_celf[i:]
    if do_greedy_simulated:
        TTE_greedy_simulated = TTE_greedy_simulated[i:]
    #TTE_CFR_heuristic = TTE_CFR_heuristic[i:]
    #Make liftup curve
    
    
    linewidth = 4
    markersize = 12
    markevery = [0,1] + list(range(4,len(node_list),4)) + [-1]
    alpha = 1
    fig_width = 6  # Width of the figure in inches
    fig_height = 5 # Height of the figure in inches

    # Create a figure with the specified size
    fig, ax = plt.subplots(figsize=(fig_width, fig_height))
    legend_labels = ['OTAPI-GA', 'OTAPI-GR', 'TARNet', 'CELF', 'SD', 'DEG', 'UB']

    if do_greedy:
        ax.plot(node_list/total_nodes, TTE_greedy/TTE_random, marker='o', linestyle='-', alpha= alpha,color='green', label='OTAPI-GR',linewidth=linewidth,markersize=markersize,markevery = markevery)
    if do_GA:
        ax.plot(node_list/total_nodes, TTE_ga/TTE_random, marker='*', linestyle='-', alpha= alpha,color='red', label='OTAPI-GA',linewidth=linewidth,markersize=markersize,markevery = markevery)
    ax.plot(node_list/total_nodes, TTE_degree/TTE_random, marker='s', linestyle='-', alpha= alpha,color='blue', label='DEG',linewidth=linewidth,markersize=markersize,markevery =markevery)
    ax.plot(node_list/total_nodes, TTE_single_discount/TTE_random, marker='p', linestyle='-', alpha= alpha, color='black', label='SD',linewidth=linewidth,markersize=markersize,markevery = markevery)
    if do_CFR:
        ax.plot(node_list/total_nodes, TTE_CFR/TTE_random, marker='v', linestyle='-', alpha= alpha, color='purple', label='TARNet',linewidth=linewidth,markersize=markersize,markevery = markevery)
    if do_CELF:
        ax.plot(node_list/total_nodes, TTE_celf/TTE_random, marker='d', linestyle='-', alpha= alpha, color='pink', label='CELF',linewidth=linewidth,markersize=markersize,markevery = markevery)
    if do_greedy_simulated:
        ax.plot(node_list/total_nodes, TTE_greedy_simulated/TTE_random, marker='P', linestyle='-', alpha= alpha, color='brown', label='UB',linewidth=linewidth,markersize=markersize,markevery=markevery)
    #plt.plot(num_nodes_500/total_nodes, compare_random_CFR_heuristic, marker='X', linestyle='-', color='grey', label='CFR Heuristic')
    #plt.title('Liftup curve (compared to random solution)')
    ax.set_xlabel(r'$k$ as \% of nodes')
    ax.set_ylabel('Liftup')
    handles, labels = plt.gca().get_legend_handles_labels()
    order = [1, 0, 4, 5, 3, 2, 6]
    order = [0,1,4,3,2]
    my_handles = [handles[idx] for idx in order]
    my_labels = [labels[idx] for idx in order]
    ax.legend(labels = my_labels,
              handles = my_handles,
               fontsize = 20, #20!
               loc = "upper right",
               bbox_to_anchor = (1.03, 1.05),
               framealpha = 0,
               labelspacing=0.3, handletextpad=0.5)
    my_path_results = "figures/" +  setting + "/"
    liftup_curve_path = my_path_results + extra + 'Liftup_curve.pdf'
    plt.savefig(liftup_curve_path, dpi=300, bbox_inches='tight',format = "pdf")
    # Show the plot
    plt.close()
    #save the solutions dicts with settings
    with open(my_path_results + 'TTE_degree.pkl', 'wb') as file:
        # Save the data to the .pkl file
        pkl.dump(TTE_degree, file)
    if do_GA:
        with open(my_path_results + 'TTE_ga.pkl', 'wb') as file:
            # Save the data to the .pkl file
            pkl.dump(TTE_ga, file)
    with open(my_path_results + 'TTE_single_discount.pkl', 'wb') as file:
        # Save the data to the .pkl file
        pkl.dump(TTE_single_discount, file)
    with open(my_path_results + 'TTE_random.pkl', 'wb') as file:
        # Save the data to the .pkl file
        pkl.dump(TTE_random, file)
    if do_CFR:
        with open(my_path_results + 'TTE_CFR.pkl', 'wb') as file:
            # Save the data to the .pkl file
            pkl.dump(TTE_CFR, file)
    if do_CELF:
        with open(my_path_results + 'TTE_celf.pkl', 'wb') as file:
            # Save the data to the .pkl file
            pkl.dump(TTE_celf, file)
    if do_greedy:
        with open(my_path_results + 'TTE_greedy.pkl', 'wb') as file:
        # Save the data to the .pkl file
            pkl.dump(TTE_greedy, file)
    if do_greedy_simulated:
        with open(my_path_results + 'TTE_greedy_simulated.pkl', 'wb') as file:
            # Save the data to the .pkl file
            pkl.dump(TTE_greedy_simulated, file)
    # with open(my_path_results + 'TTE_CFR_heuristic.pkl', 'wb') as file:
    #     # Save the data to the .pkl file
    #     pkl.dump(TTE_CFR_heuristic, file)
    # #Make plot compared to random solution
    node_list = node_list/100
    if do_greedy:
        compare_random_greedy = [greedy_solution_dict[node][2]/random_solution_dict[node][2] for node in node_list]
    if do_GA:
        compare_random_ga = [ga_solution_dict[node][2]/random_solution_dict[node][2]for node in node_list]
    compare_random_degree = [degree_solution_dict[node][2]/random_solution_dict[node][2] for node in node_list]
    compare_random_single_discount = [single_discount_solution_dict[node][2]/random_solution_dict[node][2] for node in node_list]
    if do_CFR:
        compare_random_CFR = [CFR_solution_dict[node][2]/random_solution_dict[node][2] for node in node_list]
    if do_CELF:
        compare_random_celf = [celf_solution_dict[node][2]/random_solution_dict[node][2] for node in node_list]
    if do_greedy_simulated:
        compare_random_greedy_simulated = [greedy_simulated_solution_dict[node][2]/random_solution_dict[node][2] for node in node_list]
    
    # compare_random_greedy = [(greedy_solution_dict[node][2]/random_solution_dict[node][2] -1)*100 for node in num_nodes_100]
    # compare_random_ga = [(ga_solution_dict[node][2]/random_solution_dict[node][2] -1)*100 for node in num_nodes_500]
    # compare_random_degree = [(degree_solution_dict[node][2]/random_solution_dict[node][2] -1)*100 for node in num_nodes_500]
    # compare_random_single_discount = [(single_discount_solution_dict[node][2]/random_solution_dict[node][2] -1)*100 for node in num_nodes_500]
    # compare_random_CFR = [(CFR_solution_dict[node][2]/random_solution_dict[node][2] -1)*100 for node in num_nodes_500]
    # compare_random_celf = [(celf_solution_dict[node][2]/random_solution_dict[node][2] -1)*100 for node in num_nodes_100]
    # compare_random_greedy_simulated = [(greedy_simulated_solution_dict[node][2]/random_solution_dict[node][2] -1)*100 for node in num_nodes_100]
    # compare_random_CFR_heuristic = [(CFR_heuristic_solution_dict[node][2]/random_solution_dict[node][2] -1)*100 for node in num_nodes_500]
    fig_width = 6  # Width of the figure in inches
    fig_height = 5 # Height of the figure in inches

    # Create a figure with the specified size
    fig, ax = plt.subplots(figsize=(fig_width, fig_height))
    if do_greedy:
        ax.plot(node_list/total_nodes*100, compare_random_greedy, marker='o', linestyle='-', color='green', label='Greedy (ours)',linewidth=linewidth,markersize=markersize,markevery = markevery)
    if do_GA:
        ax.plot(node_list/total_nodes*100, compare_random_ga, marker='*', linestyle='-', color='red', label='Genetic Algorithm (ours)',linewidth=linewidth,markersize=markersize,markevery = markevery)
    ax.plot(node_list/total_nodes*100, compare_random_degree, marker='s', linestyle='-', color='blue', label='Degree',linewidth=linewidth,markersize=markersize,markevery = markevery)
    ax.plot(node_list/total_nodes*100, compare_random_single_discount, marker='p', linestyle='-', color='black', label='Single Discount',linewidth=linewidth,markersize=markersize,markevery = markevery)
    if do_CFR:
        ax.plot(node_list/total_nodes*100, compare_random_CFR, marker='v', linestyle='-', color='purple', label='CFR (uplift modeling)',linewidth=linewidth,markersize=markersize,markevery = markevery)
    if do_CELF:
        ax.plot(node_list/total_nodes*100, compare_random_celf, marker='d', linestyle='-', color='pink', label='CELF',linewidth=linewidth,markersize=markersize,markevery = markevery)
    if do_greedy_simulated:
        ax.plot(node_list/total_nodes*100, compare_random_greedy_simulated, marker='P', linestyle='-', color='brown', label='Upper bound',linewidth=linewidth,markersize=markersize,markevery = markevery)
    #plt.plot(num_nodes_500/total_nodes, compare_random_CFR_heuristic, marker='X', linestyle='-', color='grey', label='CFR Heuristic')
    #plt.title('Influence spread (compared to random solution)')
    ax.set_xlabel(r'$k$ as \% of nodes')
    ax.set_ylabel('RISEO')
    #ax.legend()
    path_influence = my_path_results + extra + 'Influence_spread.pdf'
    plt.savefig(path_influence, dpi=500, bbox_inches='tight',format = "pdf")
    # Show the plot
    plt.close()
    #create matrix with each element the percentage of common selected nodes for all methods
    #We do this at 5% nodes:
    #Plot raw:
    fig_width = 6  # Width of the figure in inches
    fig_height = 5  # Height of the figure in inches

    # Create a figure with the specified size
    fig, ax = plt.subplots(figsize=(fig_width, fig_height))
    if do_greedy:
        greedy_absolute = [greedy_solution_dict[node][2] for node in node_list]
        ax.plot(node_list/total_nodes*100,greedy_absolute, marker='o', linestyle='-', color='green', label='OTAPI-GR',linewidth=linewidth,markersize=markersize,markevery = markevery)
    if do_GA:
        ga_absolute = [ga_solution_dict[node][2] for node in node_list]
        ax.plot(node_list/total_nodes*100,ga_absolute, marker='*', linestyle='-', color='red', label='OTAPI-GA',linewidth=linewidth,markersize=markersize,markevery = markevery)
    degree_absolute = [degree_solution_dict[node][2] for node in node_list]
    ax.plot(node_list/total_nodes*100,degree_absolute, marker='s', linestyle='-', color='blue', label='DEG',linewidth=linewidth,markersize=markersize,markevery = markevery)
    single_discount_absolute = [single_discount_solution_dict[node][2] for node in node_list]
    ax.plot(node_list/total_nodes*100,single_discount_absolute, marker='p', linestyle='-', color='black', label='SD',linewidth=linewidth,markersize=markersize,markevery = markevery)
    random_absolute = [random_solution_dict[node][2] for node in node_list]
    ax.plot(node_list/total_nodes*100,random_absolute, marker='x', linestyle='-', color='grey', label='Rand',linewidth=linewidth,markersize=markersize,markevery = markevery)
    if do_CFR:
        CFR_absolute = [CFR_solution_dict[node][2] for node in node_list]
        ax.plot(node_list/total_nodes*100,CFR_absolute, marker='v', linestyle='-', color='purple', label='CFR',linewidth=linewidth,markersize=markersize,markevery = markevery)
    if do_CELF:
        celf_absolute = [celf_solution_dict[node][2] for node in node_list]
        ax.plot(node_list/total_nodes*100,celf_absolute, marker='d', linestyle='-', color='pink', label='CELF',linewidth=linewidth,markersize=markersize,markevery = markevery)
    if do_greedy_simulated:
        greedy_simulated_absolute = [greedy_simulated_solution_dict[node][2] for node in node_list]
        ax.plot(node_list/total_nodes*100,greedy_simulated_absolute, marker='P', linestyle='-', color='brown', label='UB',linewidth=linewidth,markersize=markersize,markevery = markevery)
    ax.set_xlabel(r'$k$ as \% of nodes')
    ax.set_ylabel('SEO')
    # ax.legend(fontsize = 20,
    #            loc = "lower left",
    #            bbox_to_anchor = (1.03, -0.05),
    #            framealpha = 0,
    #            labelspacing=0.3, handletextpad=0.5)    
    path_influence = my_path_results + extra + 'sum_PO.pdf'
    plt.savefig(path_influence, dpi=500, bbox_inches='tight',format = "pdf")
    # Show the plot
    plt.close()
def get_similarity_matrix(total_nodes,setting,do_greedy,do_GA,
                              do_CELF,do_CFR,do_CFR_heuristic,do_random,do_greedy_simulated,do_full,my_T):
    
    #Check if the extra files exist:
    my_path = "data/allocations/" + setting + "/"
    if do_full:
        extra = "full_"
    else:
        extra = ""

    with open(my_path + extra + 'degree_solution_dict_test.pkl', 'rb') as file:
        degree_solution_dict = pkl.load(file)

    if do_GA:

        with open(my_path + extra +'ga_solution_dict_test.pkl', 'rb') as file:
            ga_solution_dict = pkl.load(file)


    with open(my_path + extra +'single_discount_solution_dict_test.pkl', 'rb') as file:
        single_discount_solution_dict = pkl.load(file)


    if do_greedy:
        with open(my_path + extra +'greedy_solution_dict_test.pkl', 'rb') as file:

            greedy_solution_dict = pkl.load(file)

    with open(my_path + extra +'random_solution_dict_test.pkl', 'rb') as file: 
        random_solution_dict = pkl.load(file)

    if do_CFR:
        with open(my_path + extra +'CFR_solution_dict_test.pkl', 'rb') as file: 
            CFR_solution_dict = pkl.load(file)
    if do_CELF:
        with open(my_path + extra +'celf_solution_dict_test.pkl', 'rb') as file:
            # Load the data from the .pkl file
            celf_solution_dict = pkl.load(file)
    if do_greedy_simulated:
        with open(my_path + extra +'greedy_simulated_solution_dict_test.pkl', 'rb') as file:
            greedy_simulated_solution_dict = pkl.load(file)
    my_path_results = "data/allocations/" + setting + "/"
    experiment_extra = "extra_experiment_"
    file_path = my_path_results + experiment_extra
    if os.path.exists(file_path+"GA_solution_dict_test.pkl"):
        with open(file_path + 'GA_solution_dict_test.pkl', 'rb') as file:
            GA_solution_dict_extra = pkl.load(file)
        ga_solution_dict = {**ga_solution_dict, **GA_solution_dict_extra}
        with open(file_path + "CFR_solution_dict_test.pkl", 'rb') as file:
            CFR_solution_dict_extra = pkl.load(file)
        CFR_solution_dict = {**CFR_solution_dict, **CFR_solution_dict_extra}
        with open(file_path + "degree_solution_dict_test.pkl", 'rb') as file:
            degree_solution_dict_extra = pkl.load(file)
        degree_solution_dict = {**degree_solution_dict, **degree_solution_dict_extra}
        with open(file_path + "single_discount_solution_dict_test.pkl", 'rb') as file:
            single_discount_solution_dict_extra = pkl.load(file)
        single_discount_solution_dict = {**single_discount_solution_dict, **single_discount_solution_dict_extra}
        with open(file_path + "random_solution_dict_test.pkl", 'rb') as file:
            random_solution_dict_extra = pkl.load(file)
        random_solution_dict = {**random_solution_dict, **random_solution_dict_extra}





    degree_T = degree_solution_dict[my_T][0].numpy()
    print(np.sum(degree_T))
    single_discount_T = single_discount_solution_dict[my_T][0].numpy()
    print(np.sum(single_discount_T))
    random_T = random_solution_dict[my_T][0].numpy()
    if do_greedy:
        greedy_indices = np.array(greedy_solution_dict[my_T][0])
        greedy_T = np.zeros(total_nodes)
        greedy_T[greedy_indices] = 1
    if do_GA:
        ga_T = np.array(ga_solution_dict[my_T][0])
    if do_CFR:
        CFR_T = CFR_solution_dict[my_T][0].numpy()
        print("CFR sum",np.sum(CFR_T))
    if do_CELF:
        celf_indices = np.array(celf_solution_dict[my_T][0])
        celf_T = np.zeros(total_nodes)
        celf_T[celf_indices] = 1
    if do_greedy_simulated:
        greedy_simulated_indices = np.array(greedy_simulated_solution_dict[my_T][0])
        greedy_simulated_T = np.zeros(total_nodes)
        greedy_simulated_T[greedy_simulated_indices] = 1
    #By multiplying two solution vectors, we get the number of common selected nodes
    num_methods = 2 + do_greedy + do_GA + do_CFR + do_CELF + do_greedy_simulated
    my_matrix = np.zeros((num_methods,num_methods))
    my_matrix[0,1] = np.sum(degree_T*single_discount_T)/my_T
    j = 1
    if do_greedy:
        j+=1
        greedy_j = j
        my_matrix[0,greedy_j] = np.sum(degree_T*greedy_T)/my_T 
    if do_GA:
        j+=1
        ga_j = j
        my_matrix[0,ga_j] = np.sum(degree_T*ga_T)/my_T
    if do_CFR:
        j+=1
        CFR_j = j
        my_matrix[0,CFR_j] = np.sum(degree_T*CFR_T)/my_T
    if do_CELF:
        j+=1
        celf_j = j
        my_matrix[0,celf_j] = np.sum(degree_T*celf_T)/my_T
    if do_greedy_simulated:
        j+=1
        greedy_simulated_j = j
        my_matrix[0,greedy_simulated_j] = np.sum(degree_T*greedy_simulated_T)/my_T
    if do_greedy:
        my_matrix[1,greedy_j] = np.sum(single_discount_T*greedy_T)/my_T
    if do_GA:
        my_matrix[1,ga_j] = np.sum(single_discount_T*ga_T)/my_T
    if do_CFR:
        my_matrix[1,CFR_j] = np.sum(single_discount_T*CFR_T)/my_T
    if do_CELF:
        my_matrix[1,celf_j] = np.sum(single_discount_T*celf_T)/my_T
    if do_greedy_simulated:
        my_matrix[1,greedy_simulated_j] = np.sum(single_discount_T*greedy_simulated_T)/my_T
    if do_greedy:
        if do_GA:
            my_matrix[greedy_j,ga_j] = np.sum(greedy_T*ga_T)/my_T
        if do_CFR:
            my_matrix[greedy_j,CFR_j] = np.sum(greedy_T*CFR_T)/my_T
        if do_CELF:
            my_matrix[greedy_j,celf_j] = np.sum(greedy_T*celf_T)/my_T
        if do_greedy_simulated:
            my_matrix[greedy_j,greedy_simulated_j] = np.sum(greedy_T*greedy_simulated_T)/my_T
    if do_GA:
        if do_CFR:
            my_matrix[ga_j,CFR_j] = np.sum(ga_T*CFR_T)/my_T
        if do_CELF:
            my_matrix[ga_j,celf_j] = np.sum(ga_T*celf_T)/my_T
        if do_greedy_simulated:
            my_matrix[ga_j,greedy_simulated_j] = np.sum(ga_T*greedy_simulated_T)/my_T
    if do_CFR:
        if do_CELF:
            my_matrix[CFR_j,celf_j] = np.sum(CFR_T*celf_T)/my_T
        if do_greedy_simulated:
            my_matrix[CFR_j,greedy_simulated_j] = np.sum(CFR_T*greedy_simulated_T)/my_T
    if do_CELF:
        if do_greedy_simulated:
            my_matrix[celf_j,greedy_simulated_j] = np.sum(celf_T*greedy_simulated_T)/my_T
    #now make the matrix symmetrical and put 1's on diagonal:
    for i in range(num_methods):
        for j in range(i):
            my_matrix[i,j] = my_matrix[j,i]
    for i in range(num_methods):
        my_matrix[i,i] = 1
    print(my_matrix)
    #write as yaml file
    matrix_dict = {"matrix": np.array(my_matrix).tolist()}
    figure_path = "figures/" + setting + "/"
    with open(figure_path +str(my_T) +'_common_selected_nodes.yaml', 'w') as file:
        # Save the data to the .pkl file
        yaml.dump(matrix_dict, file,default_flow_style=False)
    csv_path = figure_path +str(my_T)+'_common_selected_nodes.csv'
    names = ["Degree","Single Discount"]
    if do_greedy:
        names.append("Greedy")
    if do_GA:
        names.append("GA")
    if do_CFR:
        names.append("CFR")
    if do_CELF:
        names.append("CELF")
    if do_greedy_simulated:
        names.append("Greedy Simulated")
    df = pd.DataFrame(my_matrix,columns = names,index = names)
    df.to_csv(csv_path)

        
def get_degree_dist(dataset,setting):
    file = "data/simulated/" + setting + ".pkl"
    with open(file,"rb") as f:
            import pickle as pkl
            data = pkl.load(f)
    dataTest = data["test"]
    testA, testX, testT,cfTestT,POTest,cfPOTest = utils.dataTransform(dataTest,False)
    def degree_distribution(adj_matrix,setting):
        # Calculate the degree of each node
        degrees = np.sum(adj_matrix, axis=1)
        fig_width = 6  # Width of the figure in inches
        fig_height = 5 # Height of the figure in inches

        # Create a figure with the specified size
        fig, ax = plt.subplots(figsize=(fig_width, fig_height))
        # Count the frequency of each degree
        degree_count = np.bincount(degrees)
        
        # Plotting the degree distribution
        ax.bar(range(len(degree_count)), degree_count, width=0.8, color='skyblue', alpha=0.6)
        ax.set_xlabel('Degree')
        ax.set_ylabel('Frequency')
        # plt.title('Degree Distribution')
        ax.set_yscale('log')
        plt.savefig("figures/"+setting+"/degree_distribution.pdf", dpi=500, bbox_inches='tight',format = "pdf")
        plt.show()
    degree_distribution(testA.cpu().numpy().astype(int),setting)


def get_TTE_curve_total(k,dataset,setting_list,NT2O_list,do_greedy = True,do_GA=True,do_CFR=True,do_CELF=True,do_greedy_simulated=True):
    #plt.style.use("science")
    plt.rcParams['text.usetex'] = True
    plt.rcParams['pgf.texsystem'] = 'pdflatex'
    plt.rcParams['pgf.preamble'] = r'\usepackage{amsmath}'
    plt.rcParams['font.family'] = 'serif'
    plt.rcParams['font.serif'] = ['Times'] # Change to the desired serif font
    plt.rcParams['axes.labelsize'] = 26 # Change to the desired label size
    plt.rcParams['axes.titlesize'] = 20 # Change to the desired title size
    plt.rcParams['xtick.labelsize'] = 22  # Adjust size of x-axis tick labels
    plt.rcParams['ytick.labelsize'] = 22  # Adjust size of y-axis tick labels
    #plt.rcParams["mathtext.default"] = "it"
    liftup_dict = {"degree":[],"single_discount":[],"random":[],"greedy":[],"GA":[],"CFR":[],"CELF":[],"greedy_simulated":[]}
    for setting in setting_list:
        my_path = "data/allocations/" + setting + "/"
        extra_experiment = "extra_experiment_"
        #load in results:
        with open(my_path + 'degree_solution_dict_test.pkl', 'rb') as file:
            degree_solution_dict = pkl.load(file)
        if os.path.exists(my_path + extra_experiment + 'degree_solution_dict_test.pkl'):
            with open(my_path + extra_experiment + 'degree_solution_dict_test.pkl', 'rb') as file:
                degree_solution_dict_extra = pkl.load(file)
            degree_solution_dict = {**degree_solution_dict, **degree_solution_dict_extra}
        with open(my_path + 'single_discount_solution_dict_test.pkl', 'rb') as file:
            single_discount_solution_dict = pkl.load(file)
        if os.path.exists(my_path + extra_experiment + 'single_discount_solution_dict_test.pkl'):
            with open(my_path + extra_experiment + 'single_discount_solution_dict_test.pkl', 'rb') as file:
                single_discount_solution_dict_extra = pkl.load(file)
            single_discount_solution_dict = {**single_discount_solution_dict, **single_discount_solution_dict_extra}
        
        with open(my_path + 'random_solution_dict_test.pkl', 'rb') as file:
            random_solution_dict = pkl.load(file)
        if os.path.exists(my_path + extra_experiment + 'random_solution_dict_test.pkl'):
            with open(my_path + extra_experiment + 'random_solution_dict_test.pkl', 'rb') as file:
                random_solution_dict_extra = pkl.load(file)
            random_solution_dict = {**random_solution_dict, **random_solution_dict_extra}
        if do_greedy:
            with open(my_path + 'greedy_solution_dict_test.pkl', 'rb') as file:
                greedy_solution_dict = pkl.load(file)
        if do_GA:
            with open(my_path + 'ga_solution_dict_test.pkl', 'rb') as file:
                ga_solution_dict = pkl.load(file)
            if os.path.exists(my_path + extra_experiment + 'ga_solution_dict_test.pkl'):
                with open(my_path + extra_experiment + 'ga_solution_dict_test.pkl', 'rb') as file:
                    ga_solution_dict_extra = pkl.load(file)
                ga_solution_dict = {**ga_solution_dict, **ga_solution_dict_extra}
        if do_CFR:
            with open(my_path + 'CFR_solution_dict_test.pkl', 'rb') as file:
                CFR_solution_dict = pkl.load(file)
            if os.path.exists(my_path + extra_experiment + 'CFR_solution_dict_test.pkl'):
                with open(my_path + extra_experiment + 'CFR_solution_dict_test.pkl', 'rb') as file:
                    CFR_solution_dict_extra = pkl.load(file)
                CFR_solution_dict = {**CFR_solution_dict, **CFR_solution_dict_extra}
        if do_CELF:

            with open(my_path + 'celf_solution_dict_test.pkl', 'rb') as file:
                celf_solution_dict = pkl.load(file)
        if do_greedy_simulated:
            with open(my_path + 'greedy_simulated_solution_dict_test.pkl', 'rb') as file:
                greedy_simulated_solution_dict = pkl.load(file)
        #get TTE for k
        random_TTE = random_solution_dict[k][2] - random_solution_dict[0][2]
        liftup_dict["random"].append(random_TTE)
        liftup_dict["degree"].append((degree_solution_dict[k][2] - degree_solution_dict[0][2])/random_TTE)
        liftup_dict["single_discount"].append((single_discount_solution_dict[k][2] - single_discount_solution_dict[0][2])/random_TTE)
        if do_greedy:
            liftup_dict["greedy"].append((greedy_solution_dict[k][2] - greedy_solution_dict[0][2])/random_TTE)
        if do_GA:
            liftup_dict["GA"].append((ga_solution_dict[k][2] - ga_solution_dict[0][2])/random_TTE)
        if do_CFR:
            liftup_dict["CFR"].append((CFR_solution_dict[k][2] - CFR_solution_dict[0][2])/random_TTE)
        if do_CELF:
            liftup_dict["CELF"].append((celf_solution_dict[k][2] - celf_solution_dict[0][2])/random_TTE)

        if do_greedy_simulated:
            liftup_dict["greedy_simulated"].append((greedy_simulated_solution_dict[k][2] - greedy_simulated_solution_dict[0][2])/random_TTE)
    #plot the liftup curve
    linewidth = 4
    markersize = 12
    markevery = 1
    alpha = 1
    fig_width = 6  # Width of the figure in inches
    fig_height = 5 # Height of the figure in inches

    # Create a figure with the specified size
    fig, ax = plt.subplots(figsize=(fig_width, fig_height))

    if do_greedy:
        ax.plot(NT2O_list, liftup_dict['greedy'], marker='o', linestyle='-', alpha=alpha, color='green', label='OTAPI-GR', linewidth=linewidth, markersize=markersize, markevery=markevery)
    if do_GA:
        ax.plot(NT2O_list, liftup_dict['GA'], marker='*', linestyle='-', alpha=alpha, color='red', label='OTAPI-GA', linewidth=linewidth, markersize=markersize, markevery=markevery)
    
    if do_CFR:
        ax.plot(NT2O_list, liftup_dict['CFR'], marker='v', linestyle='-', alpha=alpha, color='purple', label='TARNet', linewidth=linewidth, markersize=markersize, markevery=markevery)
    ax.plot(NT2O_list, liftup_dict['degree'], marker='s', linestyle='-', alpha=alpha, color='blue', label='DEG', linewidth=linewidth, markersize=markersize, markevery=markevery)
    ax.plot(NT2O_list, liftup_dict['single_discount'], marker='p', linestyle='-', alpha=alpha, color='black', label='SD', linewidth=linewidth, markersize=markersize, markevery=markevery)
    # if do_CFR:
    #     ax.plot(NT2O_list, liftup_dict['CFR'], marker='v', linestyle='-', alpha=alpha, color='purple', label='TARNet', linewidth=linewidth, markersize=markersize, markevery=markevery)
    if do_CELF:
        ax.plot(NT2O_list, liftup_dict['CELF'], marker='d', linestyle='-', alpha=alpha, color='pink', label='CELF', linewidth=linewidth, markersize=markersize, markevery=markevery)
    if do_greedy_simulated:
        ax.plot(NT2O_list, liftup_dict['greedy_simulated'], marker='P', linestyle='-', alpha=alpha, color='brown', label='UB', linewidth=linewidth, markersize=markersize, markevery=markevery)

    # Set labels for the axes
    ax.set_xlabel(r"$\beta_{spillover}$")
    ax.set_ylabel('Liftup')
    handles, labels = plt.gca().get_legend_handles_labels()
    order = [1, 0, 2, 5, 4, 3, 6]
    my_handles = [handles[idx] for idx in order]
    my_labels = [labels[idx] for idx in order]

    ax.legend(handles= my_handles, labels=my_labels,
        fontsize=20, loc="upper left", bbox_to_anchor=(-0.03, 1.02), framealpha=0, labelspacing=0.3, handletextpad=0.5)

    # Specify the path for saving the figure
    my_path_results = "figures/Liftup" + str(k) + ".pdf"

    # Save the figure with the specified DPI, tight bounding box, and PDF format
    plt.savefig(my_path_results, dpi=300, bbox_inches='tight', format="pdf")

    # Show the plot (optional)
    plt.close()


        

