import os, codecs, math, numpy, sys, operator

class movie_predictor:

    def __init__(self):
        self.input_dir = '/Users/karthikchandrasekar/Desktop/Studies/Social_Media_Mining/SMM_H4_DEV/INPUT/ml-100k/'
        self.file_name = 'u.data'
        self.input_file = os.path.join(self.input_dir, self.file_name)

    def run_main(self):
        user_based_pred = 0
        item_based_pred = 0

        self.load_matrix()
        self.get_input_values()
        user_based_pred = self.user_based()
        print user_based_pred 

    def load_matrix(self):
        self.initialize_matrix()
        self.open_file()
        self.read_file()
        self.close_file()

    def get_input_values(self):
        self.n_size = int(sys.argv[1])
        self.user_id = int(sys.argv[2])
        self.item_id = int(sys.argv[3])

    def user_based(self):

        num = user_avg = deno = 0

        self.n_to_sim_dict = self.find_n_similar_neighbours()

        self.n_to_avg_rat_dict = self.find_user_rating_avg()

        user_avg = sum(self.user_rat_list)/float(len(self.user_rat_list))
        deno = sum(self.n_to_sim_dict.values())

        for user, sim in self.n_to_sim_dict.iteritems():
            rat_avg = self.n_to_avg_rat_dict[user]
            rate = self.user_item_matrix[user][self.item_id]
            num += sim * (rate-rat_avg)
                      
        return user_avg + (num/float(deno))


    def initialize_matrix(self):
        self.user_item_matrix = numpy.empty((2000, 2000))

    def open_file(self):
        self.fd_udata = codecs.open(self.input_file, "r", "utf-8")
        
    def read_file(self):
        for line in self.fd_udata.readlines():
            user_id, item_id, rating = line.split("\t")[:3]      
            self.user_item_matrix[int(user_id)][int(item_id)] = int(rating)

    def close_file(self):
        self.fd_udata.close()

    def find_n_similar_neighbours(self):
        row_count = 0
        user_to_simval = {}

        self.user_rat_list = self.user_item_matrix[self.user_id]

        for row in self.user_item_matrix:
            row_count += 1 
            if row_count == self.user_id:
                continue
            sim = self.cosine_similarity(self.user_rat_list, row)
            user_to_simval[row_count] = sim
        
        n_to_sim_dict = {}
        sorted_dict = sorted(user_to_simval.iteritems(), key=operator.itemgetter(1), reverse=True)           
        top_n_neighbour_tuple = sorted_dict[:self.n_size]

        for a,b in top_n_neighbour_tuple:
            n_to_sim_dict[a] = b

        return n_to_sim_dict

    def find_user_rating_avg(self):
        
        n_to_avg_rat_dict = {}
        for user in self.n_to_sim_dict.keys():
            rat_list = self.user_item_matrix[user]
            avg_rat = sum(rat_list)/float(len(rat_list))
            n_to_avg_rat_dict[user] = avg_rat

        return n_to_avg_rat_dict


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


if __name__ == "__main__":
    mp_obj = movie_predictor()
    mp_obj.run_main()
