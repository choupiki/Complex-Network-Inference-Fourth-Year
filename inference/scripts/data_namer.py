import os 

def state_point_lister(folder):

    path = "C:/Users/oscar/devel/fmri2/inference/scripts/{}".format(folder)
    state_point_list = os.listdir(path)
    #print(state_point_list)
    return state_point_list, path
