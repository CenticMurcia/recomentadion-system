
# Datos

- Pagina: https://pma.plataformaintelia.com/
- user: centic
- pass: 0D85v$P9K@@M


tabla userXprod_eventos:

| userId | prodId |     evento    | fecha |
|--------|--------|---------------|-------|
|   A    |    X   |     visita    | xxxxx |
|   A    |    X   |     visita    | xxxxx |
|   A    |    X   |     visita    | xxxxx |
|   A    |    X   | añade_carrito | xxxxx |
|   A    |    X   | quita_carrito | xxxxx |
|   A    |    X   |     compra    | xxxxx |


El peso (afinidad) usuarioXproducto puede venir dada por la siguiente formula:

afinindad = (num_vistias * 0.1) + (num_compras * 1) + (num_añade_carrito * 0.3) - (num_quita_carrito * 0.2)


tabla userXprod_pesos. Los pares (userId,prodId) deben ser unicos:


| userId | prodId | afinindad |
|--------|--------|-----------|
|   A    |    X   |    1.3    |
|   A    |    Y   |    2.7    |
|   A    |    Z   |    2.4    |
|   B    |    X   |    1.0    |
|   B    |    Z   |    0.2    |



# Funciones a implementar


### Entrenar modelo para empresa

```python
def entrenar_modelo(df_user, df_prod, df_userXprod_afinindad)
    return modelo
```

### Listar usuarios afines por producto concreto

```python
def sacar_adiencias_de_prod(modelo, prod_id)
    return lista_usuarios_afines
```

### Listar usuarios afines por producto filtros

```python
def sacar_adiencias_de_prod_nuevo(modelo, prod_criterios={categoria, subcategoria}):
    return lista_usuarios_afines
```

### Listar productos afines por usario concreto

```python
def sugerir_prods_a_usuario(modelo, user_id):
    return lista_de_productos_a_sugerir
```

### Listar productos afines por usario filtros

```python
def sugerir_prods_a_usuario_anonimo(modelo, user_criterios={edad, sexo, loaclidad})
    return lista_de_productos_a_sugerir
```

### Productos que normalmente se compran juntos

```python
def productos_sugeridos(modelo, prod_id, user_id)
    return lista_de_productos_a_sugerir
```

