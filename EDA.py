from Plots import box_plot_class, plot_empirical, corr_post_followers, time_bet_posts
import pandas as pd

df = pd.read_csv("CSVs/UsersImages.csv")

plot_empirical(df)
box_plot_class(df, "Saturation")
box_plot_class(df, "Hue")
box_plot_class(df, "Caption Length")
box_plot_class(df, "Mention Count")
box_plot_class(df, "Hashtag Count")
corr_post_followers()
time_bet_posts()
