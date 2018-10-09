import json
import numpy as np
import utils
import sys


MAX_ITEMS_PER_CHAMP = 6
EXAMPLES_PER_CHUNK = 100000
CHAMPS_PER_GAME = 10
SPELLS_PER_CHAMP = 2
SPELLS_PER_GAME = SPELLS_PER_CHAMP * CHAMPS_PER_GAME
NUM_FEATURES = CHAMPS_PER_GAME + CHAMPS_PER_GAME * SPELLS_PER_CHAMP + CHAMPS_PER_GAME * MAX_ITEMS_PER_CHAMP


class ProcessTrainingData:

    def buildNumpyDB(self):
        print("Building numpy database now. This may take a few minutes.")
        if len(sys.argv) != 3:
            print("specify absolute_ and next_ files")
            exit(-1)
        self.x_filename = sys.argv[1]
        self.y_filename = sys.argv[2]
        print("Loading input files")
        with open(self.x_filename) as f:
            with open(self.y_filename) as f1:
                self.raw_y = json.load(f1)
            self.raw_x = json.load(f)
        print("Complete")

        self.data_x_y = dict()
        print("Generating input & output vectors...")
        counter = 0
        for game_x, game_y in zip(self.raw_x, self.raw_y):
            team1_team_champs = np.array(game_x['participants'][:5])
            team2_team_champs = np.array(game_x['participants'][5:])
            converter = utils.Converter()
            team1_team_champs = [converter.champ_id2int(champ) for champ in team1_team_champs]
            team2_team_champs = [converter.champ_id2int(champ) for champ in team2_team_champs]

            # next items could be shorter than absolute items because at match end there are no next item predictions, or the losing could continue buying
            next_items = game_y['winningTeamNextItems']
            absolute_items = game_x['itemsTimeline'][:len(next_items)]

            for items_x, items_y in zip(absolute_items, next_items):
                team1_team_items_at_time_x = items_x[:5]
                team1_team_items_at_time_x = [
                    np.pad(player_items, (0, MAX_ITEMS_PER_CHAMP - len(player_items)), 'constant',
                           constant_values=(0, 0)) for player_items in team1_team_items_at_time_x]
                team1_team_items_at_time_x = np.ravel(team1_team_items_at_time_x).astype(int)

                team2_team_items_at_time_x = items_x[5:]
                team2_team_items_at_time_x = [
                    np.pad(player_items, (0, MAX_ITEMS_PER_CHAMP - len(player_items)), 'constant',
                           constant_values=(
                               0, 0)) for player_items in team2_team_items_at_time_x]
                team2_team_items_at_time_x = np.ravel(team2_team_items_at_time_x).astype(int)

                team1_team_items_at_time_x = [converter.item_id2int(int(item)) for item in team1_team_items_at_time_x]
                team2_team_items_at_time_x = [converter.item_id2int(int(item)) for item in team2_team_items_at_time_x]

                x = tuple(np.concatenate([team1_team_champs,
                                          team2_team_champs,
                                          team1_team_items_at_time_x,
                                          team2_team_items_at_time_x], 0))
                y = [converter.item_id2int(item) for item in items_y]

                # don't include dupes. happens when someone buys a potion and consumes it
                if x not in self.data_x_y:
                    self.data_x_y[x] = y

            print("current file {:.2%} processed".format(counter / len(self.raw_x)))
            counter += 1

        print("Writing to disk...")
        self.writeNextItemToNumpyFile(EXAMPLES_PER_CHUNK)

    def buildNumpyDBForPositions(self):
        print("Building numpy database now. This may take a few minutes.")
        if len(sys.argv) != 2:
            print("specify gamefile")
            exit(-1)
        self.x_filename = sys.argv[1]
        print("Loading input files")
        with open(self.x_filename) as f:
            self.raw = json.load(f)
        print("Complete")

        self.data_x = []
        print("Generating input & output vectors...")
        counter = 0
        for game in self.raw:
            team1 = np.array(game['participants'][:5])
            team2 = np.array(game['participants'][5:])
            converter = utils.Converter()

            for team in [team1, team2]:
                x = tuple([[converter.champ_id2int(participant['championId']), converter.spell_id2int(
                    participant['spell1Id']), converter.spell_id2int(
                    participant['spell2Id']), participant['kills'], participant['deaths'], participant['assists'], participant['earned'], participant['level'],
                    participant['minionsKilled'], participant['neutralMinionsKilled'], participant['wardsPlaced']] for participant in team])
                self.data_x.append(x)

            print("current file {:.2%} processed".format(counter / len(self.raw)))
            counter += 1

        print("Writing to disk...")
        self.writePositionsToNumpyFile(EXAMPLES_PER_CHUNK)

    def writeNextItemToNumpyFile(self, chunksize):

        x = list(self.data_x_y.keys())
        y = list(self.data_x_y.values())
        offset = 0
        print("Now writing numpy files to disk")
        new_file_name_x = self.x_filename[self.x_filename.rfind("/") + len('absolute_') + 1:]
        new_file_name_y = self.y_filename[self.y_filename.rfind("/") + len('next_') + 1:]
        for x_chunk, y_chunk in zip(utils.chunks(x, chunksize), utils.chunks(y, chunksize)):
            with open('training_data/next_items/processed/'+new_file_name_x+'_train_x_'+str(offset)+'.npz', "wb") as writer:
                np.savez_compressed(writer, x_chunk)
            with open('training_data/next_items/processed/'+new_file_name_y+'_train_y_'+str(offset)+'.npz', "wb") as writer:
                np.savez_compressed(writer, y_chunk)

            offset += 1
            print("{}% complete".format(int(min(100, 100*(offset*chunksize/len(x))))))

    def writePositionsToNumpyFile(self, chunksize):

        offset = 0
        print("Now writing numpy files to disk")
        new_file_name_x = self.x_filename[self.x_filename.rfind("/") + len('structuredForRole') + 1:]
        for x_chunk in utils.chunks(self.data_x, chunksize):
            with open('training_data/positions/processed/'+new_file_name_x+'_train_x_'+str(offset)+'.npz', "wb") as writer:
                np.savez_compressed(writer, x_chunk)
            offset += 1
            print("{}% complete".format(int(min(100, 100*(offset*chunksize/len(self.data_x))))))

    @staticmethod
    def _uniformShuffle(l1, l2):
        assert len(l1) == len(l2)
        rng_state = np.random.get_state()
        np.random.shuffle(l1)
        np.random.set_state(rng_state)
        np.random.shuffle(l2)

if __name__ == "__main__":
    p = ProcessTrainingData()
    p.buildNumpyDB()
