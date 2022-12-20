import numpy as np
import pandas as pd
from tqdm import tqdm
import torch





def prepare_df_pairs(df_pairs):

	#df_pairs.rename(columns={"customer_id": "user_id", "product_id": "item_id"}, inplace=True
	df_pairs["user_id"] = df_pairs["customer_id"]
	df_pairs["item_id"] = df_pairs["product_id"]

	return df_pairs


def prepare_df_users(df_users):

	### Extract user features: gender (4 columns)

	df_user_feats = pd.get_dummies(df_users.gender_id, dummy_na = True).astype("float32")  # Categorica, 1-hot encoding (4)
	return df_user_feats


def prepare_df_items(df_items):

	# Extract item features: price (1) + families (37) + features (1027) = 1065 columns

	df_item_feats = pd.concat([
		(( df_items.price - df_items.price.mean() ) / df_items.price.std() ).to_frame(), # Continua (1)
		df_items.families.str.get_dummies(','), # Categorica, N-hot encoding (37)
		df_items.features.str.get_dummies(','), # Categorica, N-hot encoding (1027)
	],axis="columns").astype("float32")

	return df_item_feats







class CollabFilteringDataset(torch.utils.data.Dataset):

	def __init__(self, df_pairs, df_users, df_items):

		self.df_pairs = prepare_df_pairs(df_pairs)
		self.df_users = prepare_df_users(df_users)
		self.df_items = prepare_df_items(df_items)
		

		#print(f"Hy un total de {self.NUM_ACTIVE_USERS} usuarios que interaccionan de un total de {len(df_users)} usuarios.")
		#print(f"Hy un total de {self.NUM_ACTIVE_ITEMS} prudcutos que interaccionan de un total de {len(df_items)} productos.")

	def __len__(self):
		return len(self.df_pairs)

	def __getitem__(self, idx):
		
		row = self.df_pairs.iloc[idx]

		return {"Item_id": row.item_id,
				"User_id": row.user_id,
				"Item_feats": self.df_items.loc[ row.item_id ].values,
				"User_feats": self.df_users.loc[ row.user_id ].values,
				"Peso": row.peso.astype("float32")}







class Model(torch.nn.Module):

	def __init__(self, df_pairs, COMMON_EMB_DIM = 32):

		super().__init__()

		self.ids_active_users,_ = torch.from_numpy(df_pairs.user_id.unique()).sort()
		self.user_id_embed_W = torch.nn.Embedding(len(self.ids_active_users), COMMON_EMB_DIM)
		self.user_id_embed_B = torch.nn.Embedding(len(self.ids_active_users), 1)
		self.user_linear_1   = torch.nn.Linear(4, COMMON_EMB_DIM) # 4 (genero)

		self.ids_active_items,_ = torch.from_numpy(df_pairs.item_id.unique()).sort()
		self.item_id_embed_W = torch.nn.Embedding(len(self.ids_active_items), COMMON_EMB_DIM)
		self.item_id_embed_B = torch.nn.Embedding(len(self.ids_active_items), 1)
		self.item_linear_1   = torch.nn.Linear(1065, 200) # 1 (precio) + 37 (familias) + 1027 (features)
		self.item_linear_2   = torch.nn.Linear(200, COMMON_EMB_DIM)

	
	##### GET embedding from user/item ID
	def encode_user_from_id(self, user_id):
		return self.user_id_embed_W( torch.searchsorted(self.ids_active_users, user_id) )

	def encode_item_from_id(self, item_id):
		return self.item_id_embed_W( torch.searchsorted(self.ids_active_items, item_id) )


	##### GET embedding from user/item features (age,geo...) (price,family...)
	def encode_user_from_feats(self, user_feats):
		return  self.user_linear_1(user_feats)

	def encode_item_from_feats(self, item_feats):
		item_emb_from_feats  = self.item_linear_1(item_feats)
		item_emb_from_feats = torch.nn.functional.relu(item_emb_from_feats)
		item_emb_from_feats = self.item_linear_2(item_emb_from_feats)
		return item_emb_from_feats


	##### FULL embedding = embedding from id + embedding from feats
	def encode_active_user(self, user_id, user_feats):
		return self.encode_user_from_id(user_id) + self.encode_user_from_feats(user_feats)

	def encode_active_item(self,  item_id, item_feats):
		return self.encode_item_from_id(item_id) + self.encode_item_from_feats(item_feats)


	##### FULL embedding for all
	def encode_user(self, user_id, user_feats):

		if user_id in self.ids_active_users:
			# Este usuario ya ha interaccionado. Su embedding sera por sus atributos y por su id
			return self.encode_active_user(user_id, user_feats)
		else:
			# Este usuario todavía no ha interaccionado. Su embedding sera solo por sus atributos
			return self.encode_user_from_feats(user_feats)

	def encode_item(self, item_id, item_feats):

		if item_id in self.ids_active_items:
			# Este producto ya ha interaccionado. Su embedding sera por sus atributos y por su id
			return self.encode_active_item(item_id, item_feats)
		else:
			# Este producto todavía no ha interaccionado. Su embedding sera solo por sus atributos
			return self.encode_item_from_feats(item_feats)


	def forward(self, x):

		user_embedding = self.encode_active_user(x["User_id"], x["User_feats"])
		item_embedding = self.encode_active_item(x["Item_id"], x["Item_feats"])

		res = (user_embedding * item_embedding).sum(axis=1) 

		res += self.user_id_embed_B( torch.searchsorted(self.ids_active_users, x["User_id"]) )[:,0] + \
		       self.item_id_embed_B( torch.searchsorted(self.ids_active_items, x["Item_id"]) )[:,0]

		return res # sigmoid_range(res, *self.y_range)






class RecSys2():


	#################################### FUNCIÓN  ####################################
	#################################### ENTRENAR #################################### 
	#################################### MODELO   ####################################

	def entrenar_modelo(self, df_pairs, df_users, df_items, valid_pct=0.2):

		BATCH_SIZE = 64
		LEARNING_RATE = 0.001
		WEIGHT_DECAY = 0.1
		EPOCHS       = 2

		print("Leyendo datos...")
		ds = CollabFilteringDataset(df_pairs, df_users, df_items)
		dl = torch.utils.data.DataLoader(ds, batch_size=BATCH_SIZE, shuffle=True)


		print("Entrenando modelo...")
		model = Model(ds.df_pairs)
		loss  = torch.nn.MSELoss()
		optim = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
		error = -1
		for epoch in range(EPOCHS):
			for batch in tqdm(dl):
				optim.zero_grad()
				model_out = model(batch)
				error = loss(model_out, batch["Peso"])
				error.backward()
				optim.step()
			print(error)


		print("Generando embeddings...")


		############################ USER EMBS

		user_ids   = torch.from_numpy( ds.df_users.index.values )
		user_feats = torch.from_numpy( ds.df_users.to_numpy() )

		# Boolean 1D tensors
		user_isActive = torch.isin(elements=user_ids, test_elements=model.ids_active_users)
		user_isNew    = ~user_isActive

		activeUser_feats      = user_feats[user_isActive]
		self.activeUser_ids   = user_ids[user_isActive]
		self.activeUser_embs  = torch.nn.functional.normalize( model.encode_active_user(self.activeUser_ids, activeUser_feats) ).detach()

		newUser_feats      = user_feats[user_isNew]
		self.newUser_ids   = user_ids[user_isNew]
		self.newUser_embs  = torch.nn.functional.normalize( model.encode_user_from_feats(newUser_feats) ).detach()


		############################ ITEM EMBS

		item_ids   = torch.from_numpy( ds.df_items.index.values )
		item_feats = torch.from_numpy( ds.df_items.to_numpy() )

		# Boolean 1D tensors
		item_isActive = torch.isin(elements=item_ids, test_elements=model.ids_active_items)
		item_isNew    = ~item_isActive

		activeItem_feats      = item_feats[item_isActive]
		self.activeItem_ids   = item_ids[item_isActive]
		self.activeItem_embs  = torch.nn.functional.normalize( model.encode_active_item(self.activeItem_ids, activeItem_feats) ).detach()

		newItem_feats      = item_feats[item_isNew]
		self.newItem_ids   = item_ids[item_isNew]
		self.newItem_embs  = torch.nn.functional.normalize( model.encode_item_from_feats(newItem_feats) ).detach()

		print("Terminado")





	#################################### FUNCIONES ####################################
	#################################### PARA USAR ####################################
	#################################### EL MODELO ####################################



	def recomendar(self,
		query_user_ids = [],
		query_item_ids = [],
		searchOn_new_users = False,
		searchOn_act_users = False,
		searchOn_new_items = False,
		searchOn_act_items = False,
		limit=0.5):


		assert limit > 0 and limit < 1


		# 1) Formar el embedding consulta (query)

		all_user_ids       = torch.cat([self.activeUser_ids,  self.newUser_ids])
		all_user_embs      = torch.cat([self.activeUser_embs, self.newUser_embs])
		selected_user_pos  = torch.isin(elements = all_user_ids, test_elements = torch.tensor(query_user_ids) )
		selected_user_embs = all_user_embs[ selected_user_pos ]

		all_item_ids       = torch.cat([self.activeItem_ids,  self.newItem_ids])
		all_item_embs      = torch.cat([self.activeItem_embs, self.newItem_embs])
		selected_item_pos  = torch.isin(elements = all_item_ids, test_elements = torch.tensor(query_item_ids) )
		selected_item_embs = all_item_embs[ selected_item_pos ]

		QUERY_EMBEDDING    = torch.cat( [selected_user_embs, selected_item_embs] ).mean(0)


		# 2) Formar los embeddings donde buscar (search on)

		SEARCHON_IDS  = []
		SEARCHON_EMBS = []

		if searchOn_new_users:
			SEARCHON_IDS.append(self.newUser_ids)
			SEARCHON_EMBS.append(self.newUser_embs)

		if searchOn_act_users:
			SEARCHON_IDS.append(self.activeUser_ids)
			SEARCHON_EMBS.append(self.activeUser_embs)

		if searchOn_new_items:
			SEARCHON_IDS.append(self.newItem_ids)
			SEARCHON_EMBS.append(self.newItem_embs)

		if searchOn_act_items:
			SEARCHON_IDS.append(self.activeItem_ids)
			SEARCHON_EMBS.append(self.activeItem_embs)

		SEARCHON_IDS  = torch.cat(SEARCHON_IDS)
		SEARCHON_EMBS = torch.cat(SEARCHON_EMBS)
		assert len(SEARCHON_EMBS)==len(SEARCHON_IDS)


		# 3) Buscar

		# 3.1. Similarity scores (pairwise cosine similarity)
		similarities = SEARCHON_EMBS @ QUERY_EMBEDDING

		# 3.2. Get good ones by threashold
		bool_matches = similarities > limit

		# 3.3. Filter by matched
		selected_ids          = SEARCHON_IDS[bool_matches]
		selected_similarities = similarities[bool_matches]

		# 3.4: sort by similarity value
		sorted_order = selected_similarities.argsort(descending=True) # Descending order positions
		return selected_ids[sorted_order]






	# USUARIO(S) --> PRODUCTOS

	def sugerirProductos_a_usuarios(self, user_ids, limit ):
		return self.recomendar(query_user_ids=user_ids, searchOn_new_items=True, searchOn_act_items=True, limit=limit)

	def sugerirProductosNuevos_a_usuarios(self, user_ids, limit):
		return self.recomendar(query_user_ids=user_ids, searchOn_new_items=True, limit=limit)

	def sugerirProductosActivos_a_usuarios(self, user_ids, limit):
		return self.recomendar(query_user_ids=user_ids, searchOn_act_items=True, limit=limit)



	# PRODUCTO(S) --> USUARIOS

	def sugerirUsuarios_para_produtos(self, item_ids, limit):
		return self.recomendar(query_item_ids=item_ids, searchOn_new_users=True, searchOn_act_users=True, limit=limit)

	def sugerirUsuariosNuevos_para_produtos(self, item_ids, limit):
		return self.recomendar(query_item_ids=item_ids, searchOn_new_users=True, limit=limit)

	def sugerirUsuariosActivos_para_produtos(self, item_ids, limit):
		return self.recomendar(query_item_ids=item_ids, searchOn_act_users=True, limit=limit)



	# PRODUCTO(S) --> PRODUCTOS Devuelve los ids de producto que normalmente se compran junto a este producto.

	def productos_parecidos(self, item_ids, limit):
		return self.recomendar(query_item_ids=item_ids, searchOn_new_items=True, searchOn_act_items=True, limit=limit)

	def productosNuevos_parecidos(self, item_ids, limit):
		return self.recomendar(query_item_ids=item_ids, searchOn_new_items=True, limit=limit)

	def productosActivos_parecidos(self, item_ids, limit):
		return self.recomendar(query_item_ids=item_ids, searchOn_act_items=True, limit=limit)





r = RecSys2()