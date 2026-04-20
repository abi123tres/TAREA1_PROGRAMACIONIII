#Abigail Cabanillas
DESARROLLO DE UNA LIBRERIA DE TENSORES EN C++

El flujo de la red consta de 8 operaciones/tareas que deben ser implementadas en el main:

1. Tensor de entrada: 1000 × 20 × 20
2. Transformación con "view": 1000 × 400
3. Multiplicación por matriz 400 × 100
4. Suma de bias 1 × 100
5. Activación ReLU
6. Multiplicación por matriz 100 × 10
7. Suma de bias 1 × 10
8. Activación Sigmoid

Estas deben ser implementadas en el main y el código debe compilar sin errores.

Para su compilación usamos el programa C++ para una mejor visualización del código, ya que es un programa extenso.
Al culminar con la ejecución debe mostrar el siguiente mensaje en la consola: "Process finished with exit code 0" para constatar que el programa compila con éxito.

