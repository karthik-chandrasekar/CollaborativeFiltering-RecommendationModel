import os, codecs, math, numpy, sys, operator
from collections import OrderedDict

class movie_predictor:

    def __init__(self):
        self.input_dir = '/Users/karthikchandrasekar/Desktop/Studies/Social_Media_Mining/SMM_H4_DEV/INPUT/ml-100k/'
        self.file_name = 'u.data'
        self.input_file = os.path.join(self.input_dir, self.file_name)

        self.m_dim = 2000
        self.n_dim = 2000

    def run_main(self):
        user_based_pred = 0
        item_based_pred = 0

        self.load_matrix(user=1)
        self.get_input_values_user_based()
        self.tweak_matrix()
        user_based_pred = self.user_based(self.user_item_matrix, self.user_id, self.item_id)
        print user_based_pred

        if self.mode == 2:
           self.user_item_matrix_orig = self.user_item_matrix 
           self.user_id_orig = self.user_id

        self.load_matrix(item=1)
        self.get_input_values_item_based()
        self.tweak_matrix()
        item_based_pred = self.user_based(self.user_item_matrix, self.user_id, self.item_id)
        print item_based_pred

        if self.mode == 2:
            self.ques_two_part_b_one(self.user_item_matrix_orig, self.user_id_orig, self.item_id)
            self.ques_two_part_b_one(self.user_item_matrix, self.user_id, self.item_id)
            

    def load_matrix(self, user=0, item=0):
        self.initialize_matrix()
        self.open_file()
        if user ==1:
            self.read_file_user_based()
        elif item ==1:
            self.read_file_item_based()
        self.close_file()

    def tweak_matrix(self):
        if self.mode == 1:
            return
        self.user_item_matrix[self.user_id][self.item_id] = -1

    def get_input_values_user_based(self):
        self.n_size = int(sys.argv[1])
        self.user_id = int(sys.argv[2])
        self.item_id = int(sys.argv[3])
        self.mode = int(sys.argv[4])

    def get_input_values_item_based(self):
        self.n_size = int(sys.argv[1])
        self.user_id = int(sys.argv[3])
        self.item_id = int(sys.argv[2])
        self.mode = int(sys.argv[4])

    def user_based(self, user_item_matrix, user_id, item_id, n_to_sim_dict={}):

        num = user_avg = deno = 0
        user_rat_list = user_item_matrix[user_id]

        if not n_to_sim_dict:
            n_to_sim_dict = self.find_n_similar_neighbours(user_item_matrix, user_rat_list)
        else:
            n_to_sim_dict = n_to_sim_dict

        n_to_avg_rat_dict = self.find_user_rating_avg(n_to_sim_dict, user_item_matrix)

        user_rat_list_ones_count = self.find_ones_count(user_rat_list) 
        user_avg = sum(user_rat_list)/float(user_rat_list_ones_count)
        deno = sum(n_to_sim_dict.values())

        for user, sim in n_to_sim_dict.iteritems():
            rat_avg = n_to_avg_rat_dict[user]
            rate = user_item_matrix[user][self.item_id]
            if rate == 0:
                continue
            num += sim * (rate-rat_avg)
                      
        return user_avg + (num/float(deno))


    def initialize_matrix(self):
        self.user_item_matrix = numpy.empty((self.m_dim, self.n_dim))

    def open_file(self):
        self.fd_udata = codecs.open(self.input_file, "r", "utf-8")
        
    def read_file_user_based(self):
        for line in self.fd_udata.readlines():
            user_id, item_id, rating = line.split("\t")[:3]      
            self.user_item_matrix[int(user_id)][int(item_id)] = int(rating)

    def read_file_item_based(self):
        for line in self.fd_udata.readlines():
            user_id, item_id, rating = line.split("\t")[:3]
            self.user_item_matrix[int(item_id)][int(user_id)] = int(rating)

    def close_file(self):
        self.fd_udata.close()

    def find_n_similar_neighbours(self, user_item_matrix, user_rat_list):
        row_count = 0
        user_to_simval = {}


        for row in user_item_matrix:
            row_count += 1 
            if row_count == self.user_id:
                continue
            sim = self.cosine_similarity(user_rat_list, row)
            user_to_simval[row_count] = sim
        
        n_to_sim_dict = OrderedDict()
        sorted_dict = sorted(user_to_simval.iteritems(), key=operator.itemgetter(1), reverse=True)           
        top_n_neighbour_tuple = sorted_dict[:self.n_size]

        for a,b in top_n_neighbour_tuple:
            n_to_sim_dict[a] = b

        return n_to_sim_dict

    def find_user_rating_avg(self, n_to_sim_dict, user_item_matrix):
        
        n_to_avg_rat_dict = {}
        for user in n_to_sim_dict.keys():
            rat_list = user_item_matrix[user]
            rat_list_ones_count = self.find_ones_count(rat_list)
            avg_rat = sum(rat_list)/float(rat_list_ones_count)
            n_to_avg_rat_dict[user] = avg_rat

        return n_to_avg_rat_dict

    def find_ones_count(self, rat_list):
        list_count = 0
        for rat in rat_list:
            if rat > 0:
                list_count +=1
        return list_count


    def cosine_similarity(self, guser_rate_list, nuser_rate_list):
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

    def ques_two_part_b_one(self, user_item_matrix, user_id, item_id):
        U, S, V = self.rank_two_approx(user_item_matrix) 
        user_item_matrix_rank = numpy.dot(U, numpy.dot(S,V))
        user_rat_list = user_item_matrix[user_id]
        n_to_sim_dict = self.find_n_similar_neighbours(user_item_matrix_rank, user_rat_list)
        print "Most similar to %s is %s" % (user_id, n_to_sim_dict.keys()[0])

        #Ques two part b second:
        n_to_avg_rat_dict = self.find_user_rating_avg(n_to_sim_dict, user_item_matrix)
        print self.user_based(user_item_matrix, user_id, item_id, n_to_sim_dict=n_to_sim_dict)
        


    def rank_two_approx(self, user_item_matrix):
        U, s_di, V = self.apply_svd(user_item_matrix)
        S = numpy.zeros((self.m_dim, self.n_dim))
        S[:self.n_dim, :self.n_dim] = numpy.diag(s_di)

        #Select first two rows of diag matrix
        S = S[:2]
        S_trans = numpy.transpose(S)
        S_trans = S_trans[:2]
        S = S_trans.transpose()        

        #Select first two rows of  V
        V = V[:2]

        #Select first two column of U
        U_trans = numpy.transpose(U)      
        U_trans = U_trans[:2]
        U = U_trans.transpose()

        return (U, S, V)
         
    def apply_svd(self, user_item_matrix):
        U, s_di, V = numpy.linalg.svd(user_item_matrix)
        return (U, s_di, V)

if __name__ == "__main__":
    mp_obj = movie_predictor()
    mp_obj.run_main()
