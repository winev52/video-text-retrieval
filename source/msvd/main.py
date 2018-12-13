import pandas as pd
import pickle

#
def main():
    raw_pd = pd.read_csv("../../data/msvd_corpus.csv")
    eng_pd = raw_pd[raw_pd["Language"] == "English"]
    eng_pd["FileName"] = eng_pd.apply(lambda x: "{}_{}_{}".format(x["VideoID"], x["Start"], x["End"]), axis=1)
    
    selected = eng_pd[["FileName", "Description"]]
    selected_np = selected.drop_duplicates().values
    with open("../../data/msvd_video_caps.pkl", "wb") as f:
        pickle.dump(selected_np, f)
    print("done")


if __name__ == "__main__":
    main()

