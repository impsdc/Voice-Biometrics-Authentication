import os 
import pickle
import glob

# deletes a registered user from database
def delete_user():
    name = input("Enter name of the user:")
    
    
    user = os.path.exists("./gmm_models/" + name + ".gmm" )
    
    if user is True:
        print('User ' + name + ' deleted successfully')
          
        [os.remove(path) for path in glob.glob('./voice_database/' + name + '/*')]
        os.removedirs('./voice_database/' + name)
        os.remove('./gmm_models/' + name + '.gmm')
        
    else:
        print('No such user !!')

if __name__ == '__main__':
    delete_user()