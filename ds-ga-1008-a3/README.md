# ds-ga-1008-a3

NYU course 2016 Spring 2016 Assignment 3

Language Modeling with Long Short Term Memory Units and Gated Recurrent Unit
============================

Training:
```bash
th main.lua --model lstm -save logs/lstm -gpu true
```
Senttence completion:
```bash
th query_senttences.lua -gpu true
```


Modifications
--------------
* Runs on CPU and GPU
* Added more comments
* Removed traces of character-level language model
