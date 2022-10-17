from fastai.tabular.all import *
from fastai.collab import *



def funcRacional_visitas(x):
    return (1 * x) / (x + 5)

def funcRacional_compras(x):
    return (5 * x) / (x + 1)

def getPeso(col_compras, col_visitias):
    return funcRacional_compras(col_compras + funcRacional_visitas(col_visitias) )



class RecSysFastai():

    #################################### CONSTRUCTOR ####################################

    def __init__(self, df, user_name="customer_id", item_name="product_id", rating_name="peso"):
        
        df[rating_name] = getPeso(df["compras"], df["visto"])

        self.df          = df
        self.user_name   = user_name
        self.item_name   = item_name
        self.rating_name = rating_name
        self.trained   = 0


    def getMemoryUsageInMegas(self):
        #a = sys.getsizeof(self)
        bytes = self.__sizeof__()
        megaBytes = bytes / ( 1000 * 1000 )
        return megaBytes


    #################################### FUNCIÓN  ####################################
    #################################### ENTRENAR #################################### 
    #################################### MODELO   ####################################

    def entrenar_modelo(self, valid_pct=0.2):
        
        dls = CollabDataLoaders.from_df(
                    ratings     = self.df,
                    valid_pct   = valid_pct,
                    user_name   = self.user_name,
                    item_name   = self.item_name,
                    rating_name = self.rating_name,
                    seed        = None,
                    bs          = 64)

        learn = collab_learner(dls, n_factors=50, y_range=(0, 5.5), metrics=[mae])
        learn.fit_one_cycle(5, 0.005, wd=0.1)

        self.items = learn.classes[learn.item]
        self.users = learn.classes[learn.user]

        self.user2idx = {user:idx for idx,user in enumerate(self.users)}
        #self.idx2user = {idx:user for idx,user in enumerate(users)}
        self.item2idx = {item:idx for idx,item in enumerate(self.items)}
        #self.idx2item = {idx:item for idx,item in enumerate(items)}

        self.user_embs = learn.u_weight.eval().cpu() # fastai.layers.Embedding
        self.item_embs = learn.i_weight.eval().cpu() # fastai.layers.Embedding
        self.user_bias = learn.u_bias.eval().cpu()   # fastai.layers.Embedding
        self.item_bias = learn.i_bias.eval().cpu()   # fastai.layers.Embedding

        # L2 Normalizaation
        self.user_embs.weight.data = F.normalize(self.user_embs.weight.data, p=2, dim=1)
        self.item_embs.weight.data = F.normalize(self.item_embs.weight.data, p=2, dim=1)

        self.trained = 1 - valid_pct




    #################################### FUNCIONES ####################################
    #################################### PARA USAR ####################################
    #################################### EL MODELO ####################################

    
    # USUARIO(S) --> PRODUCTOS
    def sugerirProdutos(self, user_ids, limit=0.3):
        """Devuelve las sugerencias (ids de producto) para un usario concreto."""
        """Devuelve las sugerencias (ids de producto) para ciertos usarios filtrados."""

        return self.emb_similaritySearch(
                        query_emb     = self.get_emb(user_ids, is_item=False),
                        searchOn_embs = self.item_embs.weight,
                        searchOn_ids  = self.items,
                        limit         = limit)


    # PRODUCTO(S) --> USUARIOS
    def adienciaUsuaros(self, item_ids, limit=0.3):
        """Devuelve la audiencia (ids de usuarios) para un producto concreto."""
        """Devuelve la audiencia (ids de usuarios) para ciertos productos filtrados."""

        return self.emb_similaritySearch(
                        query_emb     = self.get_emb(item_ids, is_item=True),
                        searchOn_embs = self.user_embs.weight,
                        searchOn_ids  = self.users,
                        limit         = limit)


    # PRODUCTO(S) --> PRODUCTOS
    def productos_parecidos(self, item_ids, limit=0.3):
        """Devuelve los ids de producto que normalmente se compran junto a este producto."""

        return self.emb_similaritySearch(
                        query_emb     = self.get_emb(item_ids, is_item=True),
                        searchOn_embs = self.item_embs.weight,
                        searchOn_ids  = self.items,
                        limit         = limit)


    # USUARIO(S) --> USUARIOS
    def usuarios_parecidos(self, user_ids, limit=0.3):
        """Devuelve los ids de producto que normalmente se compran junto a este producto."""

        return self.emb_similaritySearch(
                        query_emb     = self.get_emb(user_ids, is_item=False),
                        searchOn_embs = self.user_embs.weight,
                        searchOn_ids  = self.users,
                        limit         = limit)


    #################################### FUNCIONES  ####################################
    #################################### AUXILIARES ####################################
    #################################### PARA USAR  ####################################
    #################################### EL MODELO  ####################################

    def get_idx(self, arr, is_item=True):
        
        c2i = self.item2idx if is_item else self.user2idx

        try:
            return tensor([c2i[o] for o in arr])
        except KeyError as e:
            if is_item:
                message = f"El producto {o} no ha sido comprado o visto por ningun cliente todavía.\nPuedes ver los ids de producto disponibles con el atributo .unique_item_ids"
            else:
                message = f"El usuario {o} no ha comprado o visto por ningun producto todavía.\nPuedes ver los ids de usuario disponibles con el atributo .unique_user_ids"
            raise Exception(message)


    def get_emb(self, arr, is_item=True):
        idx   = self.get_idx(arr, is_item)
        layer = (self.item_embs if is_item else self.user_embs)
        return to_detach(layer(idx), gather=False).mean(0)



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
        sorted_order = selected_similarities.argsort(descending=True) # Descending order positions
        return selected_ids[sorted_order]
