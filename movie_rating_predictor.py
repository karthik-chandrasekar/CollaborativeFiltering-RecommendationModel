import os, codecs, math, numpy, sys, operator
from collections import OrderedDict
import matplotlib.pyplot as plt

class movie_predictor:

    def __init__(self):
        self.input_dir = '/Users/karthikchandrasekar/Desktop/Studies/Social_Media_Mining/SMM_H4_DEV/INPUT/ml-100k/'
        self.file_name = 'u.data'
        self.input_file = os.path.join(self.input_dir, self.file_name)

        self.selected_items_list = ['63', '126', '186', '55', '176', '177', '317', '356', '181', '68'] #Movielens ids for the given imdb movies

        self.selected_items_col_map = {'63':'0', '126':'1', '186':'2', '55':'3', '176':'4', '177':'5', '317':'6', '356':'7', '181':'8', '68':'9'}
        self.mode = int(sys.argv[4])

        self.col_to_selected_items_map = {'0':'63', '1':'126', '2':'186', '3':'55', '4':'176', '5':'177', '6':'317', '7':'356', '8':'181', '9':'68'}

        #Making it a square matrix for computational easiness
        
        if self.mode == 1:    
            self.m_dim = 943  
            self.n_dim = 1682
        if self.mode == 2:
            self.m_dim = 943 
            self.n_dim = 10
            
        self.rank = 2
        self.graph_drawn = 0

    def run_main(self):
        #Core function which calls several functions to find rating values with and without SVD

        #USER BASED - Ques 1 - Without SVD
        self.find_rating_without_svd(user=1)

        #Storing the user_matrix and input arguments separately as same data structures will be used for item based. 
        self.user_item_matrix_orig = self.user_item_matrix 
        self.user_id_orig = self.user_id
        self.item_id_orig = self.item_id


        #ITEM BASED - Ques 1 - Without SVD
        self.find_rating_without_svd(item=1)

        if self.mode == 1:
            return

        #USER BASED - Ques 2 - With SVD
        self.find_rating_with_svd(self.user_item_matrix_orig, self.user_id_orig, self.item_id_orig, user=1)

        #ITEM BASED - Ques 2 - With SVD
        self.find_rating_with_svd(self.user_item_matrix_trans, self.user_id, self.item_id, item=1)

            
    def find_rating_without_svd(self, user=0, item=0):
        if user ==1:
            self.load_matrix(user=1)
            self.get_input_values_user_based()
            print self.predict_rating(self.user_item_matrix, self.user_id, self.item_id)
        if item ==1:
            self.user_item_matrix_trans = self.user_item_matrix.transpose()
            self.get_input_values_item_based()
            print self.predict_rating(self.user_item_matrix_trans, self.user_id, self.item_id)
            

    def load_matrix(self, user=0, item=0):
        self.initialize_matrix()
        self.open_file()
        if user ==1:
            self.read_file_user_based()
        elif item ==1:
            self.read_file_item_based()
        self.close_file()

    def get_input_values_user_based(self):
        self.n_size = int(sys.argv[1])
        self.user_id = int(sys.argv[2]) - 1
        self.item_id = int(sys.argv[3]) - 1
        if str(self.item_id) not in self.selected_items_list:
            print "Please enter any one of these movie ids  %s " % ([int(x)+1 for x in self.selected_items_list])
            exit(0)
        self.item_id = int(self.selected_items_col_map.get(str(self.item_id), -1)) 
        self.mode = int(sys.argv[4])

    def get_input_values_item_based(self):
        self.n_size = int(sys.argv[1])
        self.user_id = int(sys.argv[3]) - 1
        self.user_id = int(self.selected_items_col_map.get(str(self.user_id), 0)) 
        self.item_id = int(sys.argv[2]) -1
        self.mode = int(sys.argv[4])

    def predict_rating(self, user_item_matrix, user_id, item_id, n_to_sim_dict={}):

        num = user_avg = deno = 0
        user_rat_list = user_item_matrix[user_id]

        if not n_to_sim_dict:
            n_to_sim_dict = self.find_n_similar_neighbours(user_item_matrix, user_rat_list, user_id)
        else:
            n_to_sim_dict = n_to_sim_dict

        n_to_avg_rat_dict = self.find_user_rating_avg(n_to_sim_dict, user_item_matrix)

        user_rat_list_ones_count = self.positive_rate_count(user_rat_list)
        if user_rat_list_ones_count !=0: 
            user_avg = sum(user_rat_list)/float(user_rat_list_ones_count)
        else:
            user_avg = 0
        
        deno = sum(n_to_sim_dict.values())

        for user, sim in n_to_sim_dict.iteritems():
            rat_avg = n_to_avg_rat_dict[user]
            rate = user_item_matrix[user][item_id]
            if rate == 0:
                continue
            num += sim * (rate-rat_avg)
                    
        if deno == 0:
            return user_avg  
        return user_avg + (num/float(deno))


    def initialize_matrix(self):
        
        self.user_item_matrix = numpy.zeros((self.m_dim, self.n_dim))


    def open_file(self):
        self.fd_udata = codecs.open(self.input_file, "r", "utf-8")
        
    def read_file_user_based(self):
        if self.mode == 1:
            for line in self.fd_udata.readlines():
                user_id, item_id, rating = line.split("\t")[:3]
                user_id = int(user_id) - 1
                item_id = int(item_id) - 1
                self.user_item_matrix[user_id][item_id] = int(rating)

        elif self.mode == 2:
            for line in self.fd_udata.readlines():
                user_id, item_id, rating = line.split("\t")[:3]
                user_id = int(user_id) - 1
                item_id = int(item_id) - 1
                if str(item_id) in self.selected_items_list:
                    item_id = int(self.selected_items_col_map.get(str(item_id)))
                    self.user_item_matrix[user_id][item_id] = int(rating)
                    

    def read_file_item_based(self):
    
        if self.mode == 1:
            for line in self.fd_udata.readlines():
                user_id, item_id, rating = line.split("\t")[:3]
                self.user_item_matrix[int(item_id)][int(user_id)] = int(rating)

        elif self.mode ==2:
            for line in self.fd_udata.readlines():
                user_id, item_id, rating = line.split("\t")[:3]
                if item_id in self.selected_items_list:
                    self.user_item_matrix[int(item_id)][int(user_id)] = int(rating)
            

    def close_file(self):
        self.fd_udata.close()

    def find_n_similar_neighbours(self, user_item_matrix, user_rat_list, user_id):

        #Finds top n similar neighbours
        row_count = 0
        user_to_simval = {}

        for row in user_item_matrix:
            if row_count == user_id:
                row_count += 1 
                continue
            sim = self.cosine_similarity(user_rat_list, row)
            user_to_simval[row_count] = sim
            row_count += 1 

        n_to_sim_dict = OrderedDict()
        sorted_dict = sorted(user_to_simval.iteritems(), key=operator.itemgetter(1), reverse=True)           
        top_n_neighbour_tuple = sorted_dict[:self.n_size]

        for a,b in top_n_neighbour_tuple:
            n_to_sim_dict[a] = b

        return n_to_sim_dict

    def find_user_rating_avg(self, n_to_sim_dict, user_item_matrix):
       
        #Finds avg user rating  
        n_to_avg_rat_dict = {}
        for user in n_to_sim_dict.keys():
            rat_list = user_item_matrix[user]
            rat_list_ones_count = self.positive_rate_count(rat_list)
            if rat_list_ones_count != 0:
                avg_rat = sum(rat_list)/float(rat_list_ones_count)
            else:
                avg_rat = 0
            n_to_avg_rat_dict[user] = avg_rat

        return n_to_avg_rat_dict

    def positive_rate_count(self, rat_list):

        #Finds number of one's in the given list
        list_count = 0
        for rat in rat_list:
            if rat > 0:
                list_count +=1
        return list_count


    def cosine_similarity(self, guser_rate_list, nuser_rate_list):

        #Finds consine similarity for given two items

        d1 = self.root_square_sum(guser_rate_list)
        d2 = self.root_square_sum(nuser_rate_list)
        d = 0.0       
 
        n = self.cosine_sim_num(guser_rate_list, nuser_rate_list)
        d = d1 * d2 
            
        if d ==0 :
            return 0           

        return n/d

    def root_square_sum(self, user_rate_list):
        tot_val = 0
        for rate in user_rate_list:
            tot_val += rate*rate

        return math.sqrt(tot_val)  
                
    def cosine_sim_num(self, guser_rate_list, nuser_rate_list):
        tot_val = 0
        for a in range(len(guser_rate_list)):
            if a == self.item_id:
                continue
            tot_val += guser_rate_list[a] * nuser_rate_list[a]
        return tot_val

    def find_rating_with_svd(self, user_item_matrix, user_id, item_id, user=0, item=0):

        #Perform rank approximization, find similartiy using this matrix and use it for predicting rate.

        user_item_matrix[user_id][item_id] = -1
        U, S, V = self.rank_two_approx(user_item_matrix)

        if not self.graph_drawn:
            self.plot_graph_svd(U, V) 

        user_item_matrix_rank = numpy.dot(U, numpy.dot(S,V))
        user_rat_list = user_item_matrix[user_id]
        n_to_sim_dict = self.find_n_similar_neighbours(user_item_matrix_rank, user_rat_list, user_id)

        if user:
            print "Most similar to user %s is user %s" % (user_id+1, n_to_sim_dict.keys()[0]+1)
        if item:
            print "Most similar to item %s is item %s" % (int(self.col_to_selected_items_map.get(str(user_id))) +1, int(self.col_to_selected_items_map.get(str(n_to_sim_dict.keys()[0])))+1)

        print self.predict_rating(user_item_matrix, user_id, item_id, n_to_sim_dict=n_to_sim_dict)


    def plot_graph_svd(self, U, V):
        self.graph_drawn = 1
        self.plot_graph_user(U, 'user')
        self.plot_graph_item(V, 'item')

          
    def plot_graph_user(self, U, fname):
        x_points_list = []
        y_points_list = []
        row_count = 0

        for row in U:
            x_point = row[0]
            y_point = row[1]
            if not x_point and not y_point:
                row_count += 1
                continue
            plt.plot(x_point, y_point, 'ro')
            plt.text(x_point, y_point, str(row_count+1), fontsize=5)
            row_count += 1
        plt.savefig(fname)
        plt.close()

    def plot_graph_item(self, V, fname):
        x_points_list = []
        y_points_list = []
        row_count =0

        V_trans = V.transpose()
        for row in V_trans:
            x_point = row[0]
            y_point = row[1]
            if not x_point and not y_point:
                row_count += 1
                continue
            plt.plot(x_point, y_point, 'ro')
            plt.text(x_point, y_point, int(self.col_to_selected_items_map.get(str(row_count)))+1, fontsize=12)
            row_count += 1
        plt.savefig(fname)
        plt.close()

    def rank_two_approx(self, user_item_matrix):
        
        #Performs rank approximation on the matrix
        U, s_di, V = self.apply_svd(user_item_matrix)
        S = numpy.zeros((self.m_dim, self.n_dim))
        S[:self.n_dim, :self.n_dim] = numpy.diag(s_di)

        #Select first two diag elements  of diag matrix
        S = S[:self.rank]
        S_trans = numpy.transpose(S)
        S_trans = S_trans[:self.rank]
        S = S_trans.transpose()        

        #Select first two rows of  V
        V = V[:self.rank]

        #Select first two column of U
        U_trans = numpy.transpose(U)      
        U_trans = U_trans[:self.rank]
        U = U_trans.transpose()

        return (U, S, V)
         
    def apply_svd(self, user_item_matrix):
    
        U, s_di, V = numpy.linalg.svd(user_item_matrix)
        return (U, s_di, V)

if __name__ == "__main__":
    mp_obj = movie_predictor()
    mp_obj.run_main()
