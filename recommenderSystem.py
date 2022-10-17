import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import optuna

class RecSys():

    #################################### CONSTRUCTOR ####################################

    def __init__(self, df_pairs, df_users=None, df_items=None, userName="customer_id", itemName="product_id", scoreName="peso"):
        
        self.trained   = 0
        self.user_embs = None
        self.item_embs = None

        ###################################### Save user-item pairs dataframe
        samples_user_id = df_pairs[userName].values
        samples_item_id = df_pairs[itemName].values
        samples_scores  = df_pairs[scoreName].values
        
        self.unique_user_ids, samples_user_iid = np.unique( samples_user_id, return_inverse=True)
        self.unique_item_ids, samples_item_iid = np.unique( samples_item_id, return_inverse=True)

        self.num_users = len(self.unique_user_ids)
        self.num_items = len(self.unique_item_ids)

        self.user_id_2_iid = lambda user_id: self.unique_user_ids.tolist().index(user_id)
        self.item_id_2_iid = lambda item_id: self.unique_item_ids.tolist().index(item_id)
        #self.user_iids_2_ids = lambda user_iids: self.unique_user_ids[ user_iids ]
        #self.item_iids_2_ids = lambda item_iids: self.unique_item_ids[ item_iids ]
        
        # self.samples is a Numpy record array: https://numpy.org/doc/stable/user/basics.rec.html#record-arrays
        self.samples = np.rec.fromarrays((samples_user_iid, samples_item_iid, samples_scores), names=('user', 'item', "score"))
        
        self.b = self.samples.score.mean()
        self.m = self.samples.score.max()
        
        ########################### Save user and item extra information
        self.df_users = df_users
        self.df_items = df_items
        
        # TODO: Comprobar que los df_pairs.user_id esten en df_users.id
        # TODO: Comprobar que los df_pairs.item_id esten en df_items.id


    def getMemoryUsageInMegas(self):
        #a = sys.getsizeof(self)
        bytes = self.__sizeof__()
        megaBytes = bytes / ( 1000 * 1000 )
        return megaBytes




    #################################### FUNCIÓN  ####################################
    #################################### ENTRENAR #################################### 
    #################################### MODELO   ####################################

    def entrenar_modelo_automaticamente(self, num_de_pruebas=20, train_perc=0.8):
        """
        Función AUTOMÁTICA para entrenar el modelo.

        Mediante la búsqueda de hiperparemetros (librería Optuna)
        hace varias pruebas (num_de_pruebas) para determinar empíricamente
        cuáles son los mejores hiperparámetros para entrenar el modelo
        para los datos de una compañía.

        Las pruebas usan el 80% de los datos para entrenar
        y el 20% de los datos para validar (evaluar) el entrenamiento.

        Los hiperparámetros a probar son:
        - Tamaño del embedding (emb_size):  Será un valor entre 10 y 50
        - Velocidad de aprendizaje (lr):    Sera un valor entre 0.0001 y 0.1
        - Número de iteraciones (epochs):   Será un valor entre 5 y 25
        - Regularización weight decay (wd): Sera un valor entre 0.0001 y 0.1

        Se recomienda realizar al menos 20 pruebas de entrenamiento
        para "encontrar" empiricamente los mejores hiperparámetros.
        Realizar más de 20 pruebas es mejor aún, pero tarda mas.

        Una vez encontrados los mejores hiperparámetros, éstos son usados
        para realizar un último entrenamiento con el 100% de los datos
        y generar el modelo final.
        """

        ################### PARTE 1: Hacer pruebas para encontrar los mejores hiperparametros

        # Separate training and validation samples
        split_idx = int(len(self.samples) * train_perc)
        np.random.shuffle(self.samples)
        train_samples = self.samples[:split_idx]
        valid_samples = self.samples[split_idx:]

        def func_a_minimizar(trial):
            train_log = self.entrenar_modelo_manualmente(
                train_samples = train_samples,
                valid_samples = valid_samples,
                embSize   = trial.suggest_int('embSize', 10, 50),
                lr        = trial.suggest_float("lr", 1e-5, 1, log=True),
                wd        = trial.suggest_float("wd", 1e-5, 1, log=True),
                epochs    = trial.suggest_int('epochs', 5, 25))
            return train_log["valid_mse"][-1]

        study = optuna.create_study()
        study.optimize(func_a_minimizar, n_trials=num_de_pruebas)
        study.best_params
        # study.best_params =~ {'embSize': 16, 'lr': 0.00237, ...}


        ################### PARTE 2: Entrenar con todos los datos con los mejores hiperparametros

        # Usar todos los datos para entrenar
        # No perder ningún dato para validar
        # Usar los mejores hiperparámetros

        self.entrenar_modelo_manualmente(
                train_samples = self.samples,
                valid_samples = None,
                **study.best_params)

        self.trained = 1




    #################################### FUNCIONES ####################################
    #################################### PARA USAR ####################################
    #################################### EL MODELO ####################################

    def sugerirProds_a_usuarioConcreto(self, user_id, limit=0.3):
        """Devuelve las sugerencias (ids de producto) para un usario concreto."""

        return self.emb_similaritySearch(
                        query_emb     = self.get_user_emb(user_id),
                        searchOn_embs = self.item_embs,
                        searchOn_ids  = self.unique_item_ids,
                        limit         = limit)

    def sugerirProds_a_usuariosFiltrados(self, user_ids, limit=0.3):
        """Devuelve las sugerencias (ids de producto) para ciertos usarios filtrados."""

        return self.emb_similaritySearch(
                        query_emb     = self.get_users_avgEmb(users_ids),
                        searchOn_embs = self.item_embs,
                        searchOn_ids  = self.unique_item_ids,
                        limit         = limit)

    def adienciaUsuaros_de_productoConcreto(self, item_id, limit=0.3):
        """Devuelve la audiencia (ids de usuarios) para un producto concreto."""

        return self.emb_similaritySearch(
                        query_emb     = self.get_item_emb(item_id),
                        searchOn_embs = self.user_embs,
                        searchOn_ids  = self.unique_user_ids,
                        limit         = limit)

    def adienciaUsuaros_de_productosFiltrados(self, item_ids, limit=0.3):
        """Devuelve la audiencia (ids de usuarios) para ciertos productos filtrados."""

        return self.emb_similaritySearch(
                        query_emb     = self.get_items_avgEmb(items_ids),
                        searchOn_embs = self.user_embs,
                        searchOn_ids  = self.unique_user_ids,
                        limit         = limit)


    def productos_parecidos(self, item_id, limit=0.3):
        """Devuelve los ids de producto que normalmente se compran junto a este producto."""

        return self.emb_similaritySearch(
                        query_emb     = get_item_emb(item_id),
                        searchOn_embs = self.item_embs,
                        searchOn_ids  = self.unique_item_ids,
                        limit         = limit)


    #################################### FUNCIONES  ####################################
    #################################### AUXILIARES ####################################
    #################################### PARA USAR  ####################################
    #################################### EL MODELO  ####################################

    def get_user_emb(self, user_id):
        
        if user_id in self.unique_user_ids:
            user_iid = self.user_id_2_iid( user_id ) # 1) Convert real user id to internal user id
            user_emb = self.user_embs[ user_iid ]    # 2) Get user embedding
            return user_emb
        else:
            raise Exception(f"El usuario {user_id} no ha comprado o visto por ningun producto todavía.\n"
                             "Puedes ver los ids de usuario disponibles con el atributo .unique_user_ids")


    def get_item_emb(self, item_id):

        if item_id in self.unique_item_ids:
            item_iid = self.item_id_2_iid( item_id ) # 1) Convert real item id to internal item id
            item_emb = self.item_embs[ item_iid ]    # 2) Get item id embedding
            return item_emb
        else:
            raise Exception(f"El producto {item_id} no ha sido comprado o visto por ningun cliente todavía.\n"
                             "Puedes ver los ids de producto disponibles con el atributo .unique_item_ids")
   

    def get_users_avgEmb(self, users_ids):
        users_embs   = [ self.get_user_emb(user_id) for user_id in users_ids ]
        users_embs   = np.stack(users_embs)
        users_avgEmb = users_embs.mean(axis=0)
        users_avgEmb = users_avgEmb / np.linalg.norm(users_avgEmb) # L2 norm
        return users_avgEmb


    def get_items_avgEmb(self, items_ids):
        items_embs   = [ self.get_item_emb(item_id) for item_id in items_ids ]
        items_embs   = np.stack(items_embs)
        items_avgEmb = items_embs.mean(axis=0)
        items_avgEmb = items_avgEmb / np.linalg.norm(items_avgEmb) # L2 norm
        return items_avgEmb


    def emb_similaritySearch(self, query_emb, searchOn_embs, searchOn_ids, limit=0.3):

        if self.trained == 0:
            print("ERROR:")
            print("El modelo todavía no se ha entrenado. Por favor llama al metodo train()")
            print()
            return
        elif self.trained > 0 and self.trained < 1:
            print("WARNING:")
            print(f"El modelo ha sido entrenado sólo con el {self.trained * 100}% de los datos.")
            print("Se recomienda reentrenar al modelo con el 100% de los datos")
            print("con parametros que eviten en sobreajuste (overfitting).")
            print("Por favor, reentrena llamando al metodo train(train_perc=1)")
            print()

        if query_emb     is None: return
        if searchOn_embs is None: return
            
        assert limit > 0 and limit < 1
        assert len(searchOn_embs)==len(searchOn_ids)


        # 1. Similarity scores (pairwise cosine similarity)
        similarities = searchOn_embs @ query_emb

        # 2. Get good ones by threashold
        bool_matches = similarities > limit

        # 3. Filter by matched
        selected_ids          = searchOn_ids[bool_matches]
        selected_similarities = similarities[bool_matches]

        # 4: sort by similarity value
        sorted_order = selected_similarities.argsort()[::-1] # Descending order positions
        return selected_ids[sorted_order]



    #################################### FUNCIONES  ####################################
    #################################### AUXILIARES ####################################
    #################################### ENTRENAR   #################################### 
    #################################### MODELO     ####################################


    def entrenar_modelo_manualmente(self, train_samples, valid_samples, embSize, lr, wd, epochs):
        """
        Función MANUAL para entrenar el modelo.

        Perform matrix factorization to predict empty entries in a matrix.

        Arguments
        - embSize (int)    : number of latent dimensions
        - lr (float)       : learning rate
        - wd (float)       : regularization parameter
        """
        
        # Initialize user and item embeddings
        self.user_embs = np.random.normal(scale=1./embSize, size=(self.num_users, embSize))
        self.item_embs = np.random.normal(scale=1./embSize, size=(self.num_items, embSize))

        # Initialize user and item biases
        self.b_u = np.zeros(self.num_users)
        self.b_i = np.zeros(self.num_items)

        # Perform stochastic gradient descent for number of epochs
 

        if valid_samples is None:
            train_log = np.array([], dtype=[('epoch', "u1"), ('train_mae', "f"), ('train_mse', "f")])
        else:
            train_log = np.array([], dtype=[('epoch', "u1"), ('train_mae', "f"), ('train_mse', "f"), ('valid_mae', "f"), ('valid_mse', "f")])


        for i in range(epochs):

            # TRAIN
            self.sgd_epoch(train_samples, lr, wd)

            if valid_samples is None:
                # NOT VALIDATE
                train_mae, train_mse = self.error(train_samples)
                train_log = np.append(train_log, np.array([(i+1,train_mae,train_mse)], dtype=train_log.dtype))
                print("Epoch: %d ; trMAE = %.4f trMSE = %.4f" % (i+1, train_mae, train_mse))
            else:
                # YES VALIDATE
                train_mae, train_mse = self.error(train_samples)
                valid_mae, valid_mse = self.error(valid_samples)
                train_log = np.append(train_log, np.array([(i+1,train_mae,train_mse,valid_mae,valid_mse)], dtype=train_log.dtype))
                print("Epoch: %d ; trMAE = %.4f trMSE = %.4f valMAE = %.4f valMSE = %.4f" % (i+1, train_mae, train_mse, valid_mae, valid_mse))

        return train_log


    def sgd_epoch(self, train_samples, lr, wd):
        """
        Perform stochastic graident descent epoch
        """
        np.random.shuffle(train_samples) # This also shuffles self.samples !!!

        for user, item, groudTruth in train_samples:

            # Computer prediction and error
            prediction = self.get_prediction(user, item)
            err = (prediction - groudTruth)

            # Update biases
            #                       grad  __weight_decay__
            self.b_u[user] -= lr * (err + wd*self.b_u[user])
            self.b_i[item] -= lr * (err + wd*self.b_i[item])

            # Update user and item embeddings
            #                                ___________grad______________   _______weight_decay______
            self.user_embs[user, :] -= lr * (err * self.item_embs[item, :] + wd*self.user_embs[user,:])
            self.item_embs[item, :] -= lr * (err * self.user_embs[user, :] + wd*self.item_embs[item,:])


    def error(self, samples):
        """
        A function to compute the errors:
        - MAE: Mean Absoulte Error
        - MSE: Mean Squared Error
        """
        mae = 0
        mse = 0
        for user, item, real in samples:
            diference = real - self.get_prediction(user, item)
            mae += abs(diference)
            mse += pow(diference, 2)
        mae /= len(samples)
        mse /= len(samples)
        
        return mae, mse


    def plot_training(self, train_log):
        self.plot_metric(train_log, metric="mae", title="Mean Absolute Error")
        self.plot_metric(train_log, metric="mse", title="Mean Square Error")


    def plot_metric(self, train_log, metric, title):
        plt.figure(figsize=((16,4)))
        plt.tick_params(labelright=True)
        plt.plot(train_log["epoch"], train_log["train_"+metric])
        plt.plot(train_log["epoch"], train_log["valid_"+metric], linewidth=3)
        plt.xticks(train_log["epoch"], train_log["epoch"])
        plt.xlabel("Epoch")
        #plt.ylabel(metric)
        plt.title(title)
        plt.grid(axis="y")
        plt.show()


    def get_prediction(self, u, i):
        """
        Get the predicted score of user u and item i
        """
        return self.b + self.b_u[u] + self.b_i[i] + self.user_embs[u, :].dot(self.item_embs[i, :].T)
        #return self.m * ( self.b_u[u] + self.b_i[i] + self.user_embs[u, :].dot(self.item_embs[i, :].T) )

    def full_matrix(self):
        """
        Compute the full matrix:
        Matrix product of embedding, plus broadcasting biases.
        """
        return self.b + self.b_u[:,np.newaxis] + self.b_i + (self.user_embs @ self.item_embs.T)
    

    def draw_matrix(self, margin=True):
        
        print("Number of users:", self.num_users)
        print("Number of items:", self.num_items)
        print("Number of interactions:", len(self.samples))
                          
        if margin and (self.df_users is not None) and (self.df_items is not None):
            matrix = np.full( shape=(len(self.df_users), len(self.df_items)), fill_value=127, dtype=np.uint8 )
            matrix[:self.num_users, :self.num_items] = 0
        else:
            matrix = np.full( shape=(self.num_users, self.num_items), fill_value=0, dtype=np.uint8 )
            
        for user_idx, item_idx, score in self.samples:
            matrix[user_idx, item_idx] = 255
    
        display(Image.fromarray(matrix))
        
        plt.figure(figsize=((4,2)))
        plt.title("Variable a predecir")
        plt.hist(self.samples.score, bins=20)
        plt.show()


    def l2_norm(self, emb_matrix):
        norms = np.sqrt( np.sum( emb_matrix**2, axis=1) )
        norms = np.clip( norms, a_min=1e-10, a_max=np.inf ) # Avoid dividing by zero
        #norms = np.linalg.norm(x, axis=1) # Otra forma de calcular los norms
        return emb_matrix / norms[:,np.newaxis]