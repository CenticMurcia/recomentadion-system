





"""

tabla userXprod_eventos:

| userId | prodId |     evento    | fecha |
|--------|--------|---------------|-------|
|	A    |    X   |     visita    | xxxxx |
|	A    |    X   |     visita    | xxxxx |
|	A    |    X   |     visita    | xxxxx |
|	A    |    X   | añade_carrito | xxxxx |
|	A    |    X   | quita_carrito | xxxxx |
|	A    |    X   |     compra    | xxxxx |


la afinidad usuarioXproducto
puede venir dada por la siguiente formula

afinindad = (num_vistias * 0.1) + (num_compras * 1) + (num_añade_carrito * 0.3) - (num_quita_carrito * 0.2)



tabla userXprod_afinindad:

los pares (userId,prodId) deben ser unicos

| userId | prodId | afinindad |
|--------|--------|-----------|
|   A    |    X   |    1.3    |
|   A    |    Y   |    2.7    |
|   A    |    Z   |    2.4    |
|   B    |    X   |    1.0    |
|   B    |    Z   |    0.2    |
"""

def entrenar_modelo(df_user, df_prod, df_userXprod_afinindad)
    return modelo


modelo_empresa_A = entrenar_modelo(df_user, df_prod, df_userXprod_afinindad)
modelo_empresa_B = entrenar_modelo(df_user, df_prod, df_userXprod_afinindad)
modelo_empresa_C = entrenar_modelo(df_user, df_prod, df_userXprod_afinindad)



=======================================

def sacar_adiencias_de_prod(modelo, prod_id)
	return lista_usuarios_afines

def sacar_adiencias_de_prod_nuevo(modelo, prod_criterios={categoria, subcategoria}):
	return lista_usuarios_afines

-------------

def sugerir_prods_a_usuario(modelo, user_id):
	return lista_de_productos_a_sugerir

def sugerir_prods_a_usuario_anonimo(modelo, user_criterios={edad, sexo, loaclidad})
	return lista_de_productos_a_sugerir

-------------

# Productos que normalmente se compran juntos
def productos_sugeridos(modelo, prod_id, user_id)
	return lista_de_productos_a_sugerir


