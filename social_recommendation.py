import os, codecs, math, numpy, sys, operator


class social_reco:
    def __init__(self):
        self.selected_items_list = [64, 127, 187, 56, 177, 178, 318, 357, 182, 69] #Movielens ids for the given imdb movies
        self.input_dir = '/Users/karthikchandrasekar/Desktop/Studies/Social_Media_Mining/SMM_H4_DEV/INPUT/ml-100k/'
        self.file_name = 'u.data'
        self.input_file = os.path.join(self.input_dir, self.file_name)

        self.m_dim = 2000
        self.n_dim = 2000     

    def run_main(self):
        self.part_one()


    def part_one(self):
        user_based_pred = 0

        self.load_matrix(user=1)
        U, S, V = self.rank_two_approx()
        self.plot_matrix(U,S,V)

    def load_matrix(self, user=0, item=0):
        self.initialize_matrix()
        self.open_file()
        if user ==1:
            self.read_file_user_based()
        self.close_file()

    def rank_two_approx(self):
        U, s_di, V = self.apply_svd()
        S = numpy.zeros((self.m_dim, self.n_dim))
        S[:self.n_dim, :self.n_dim] = numpy.diag(s_di)

        #Select first two rows of diag matrix
        S = S[:2]

        #Select first two rows of  V
        V = V[:2]

        #Select first two column of U
        U_trans = numpy.transpose(U)      
        U_trans = U_trans[:2]
        U = U_trans.transpose()

        return (U, S, V)
         
    def plot_matrix(self, U, S, V):
        pass

    def apply_svd(self):
        U, s_di, V = numpy.linalg.svd(self.user_item_matrix)
        return (U, s_di, V)

    def initialize_matrix(self):
        self.user_item_matrix = numpy.empty((self.m_dim, self.n_dim))
    
    def open_file(self):
        self.fd_udata = codecs.open(self.input_file, "r", "utf-8")

    def read_file_user_based(self):
        for line in self.fd_udata.readlines():
            user_id, item_id, rating = line.split("\t")[:3]
            if item_id in self.selected_items_list:      
                self.user_item_matrix[int(user_id)][int(item_id)] = int(rating)
            else:
                self.user_item_matrix[int(user_id)][int(item_id)] = -1

    def close_file(self):
        self.fd_udata.close()
                
if __name__ == "__main__":
    sr_obj = social_reco()
    sr_obj.run_main()
