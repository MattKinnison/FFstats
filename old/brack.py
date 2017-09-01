def seeding(num):

    brack = [[1]]
    for seed in range(2,num+1):
        #split
        unpaired = [game for game in brack if len(game) == 1]
        if len(unpaired) == 0:
             new_brack = []
             for game in brack:
                 new_brack.append([game[0]])
                 new_brack.append([game[1]])
             brack = new_brack
        #insert
        in_game = max([game for game in brack if len(game) == 1])
        brack = [game if game != in_game else [in_game[0],seed] for game in brack]

    return brack


