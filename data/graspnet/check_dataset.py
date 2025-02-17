from graspnetAPI import GraspNet

# GraspNetAPI example for checking the data completeness.
# change the graspnet_root path

if __name__ == '__main__':

    ####################################################################
    graspnet_root = '/home/clark/ex_storage/graspnet_dataset'  ### ROOT PATH FOR GRASPNET ###
    ####################################################################

    g = GraspNet(graspnet_root, 'kinect', 'all')
    if g.checkDataCompleteness():
        print('Check for kinect passed')


    g = GraspNet(graspnet_root, 'realsense', 'all')
    if g.checkDataCompleteness():
        print('Check for realsense passed')

