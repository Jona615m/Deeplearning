Fixed errors in assignment 1.1 (Frederik):

- Vi kaldte compute_distances i predict(), men den fandtes ikke. Den hed calculate_distance, som jeg har ændret til calculate_distances
- Vi havde et overflow problemer ift calculate_distances. Det skulle konverteres til float. Det var også derfor vi fik så lav accuracy
- Vi manglede i hyperparameter tuning at plotte det til den graf som han viser.
