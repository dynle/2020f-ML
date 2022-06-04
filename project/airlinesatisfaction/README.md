# Trends of Covid-19, Airport Traffic in the US, Passenger Satisfaction

This research firstly deals with trends of Covid-19 new cases and airport traffic in the US in order to examine traffic trends in the near future and predict the time when passengers start to use airlines as like before the outbreak of Covid-19. And then, the order from the most to worst importance for affecting passenger satisfaction on airline experience is found through data of passenger satisfaction survey.

Check [this](./Final_report_Doyoon_Lee.pdf) (`Final_report_Doyoon_lee.pdf`) to read the full version of paper

## Tech
* Python
* [Scikit-learn](https://scikit-learn.org/stable/)
* BeautifulSoup
* Pandas
* Numpy
* Matplotlib
* Seaborn

## US Trends of Covid-19 New Cases as of 2020

<img width="650" alt="Screen Shot 2022-06-04 at 4 18 37 PM" src="https://user-images.githubusercontent.com/36508771/171989422-7925d94e-2277-4d49-8b3b-b9f68d8be6e8.png">

## US Trends of Airport Traffic as of 2020

<img width="664" alt="Screen Shot 2022-06-04 at 4 20 56 PM" src="https://user-images.githubusercontent.com/36508771/171989430-3d695a92-c47f-4b37-baf1-46cf35604a62.png">

## Comparison among Multiple Ensemble Models

<img width="751" alt="Screen Shot 2022-06-04 at 4 22 11 PM" src="https://user-images.githubusercontent.com/36508771/171989440-1043f53a-09b8-4c2a-80c5-33d3cd556582.png">

## Features Importances

<img width="872" alt="Screen Shot 2022-06-04 at 4 23 18 PM" src="https://user-images.githubusercontent.com/36508771/171989449-ad5ae081-5fc0-4fcd-99b0-910175822253.png">

## Conclusion
The traffic trend calculated by polyfit( ) and poly1d( ) functions shows the trend line keeps go up and recovers the traffic before affected by Covid-19 from current situation in only 34 days. The increasing trend is reliable to some extent, but such the rapid increasing rate is suspicious therefore it would be better to use other prediction techniques related with deep learning in order to predict the future traffic more precisely.

From the trends of number of new cases and passengers calculated by numpy, the traffic would increase regardless of increasing new cases for now. The increasing rate of the traffic would even more rapidly rise if vaccine efficacy works in the US where has started to provide the vaccine from December 14. Therefore, airline companies have to improve their services as soon as possible to prepare for attracting potential passengers.

The priority of upgrading their services is from the highest feature importance for passenger satisfaction calculated by randomforestclassifier machine learning model which has the highest accuracy: ‘online boarding’ (0.156). It is followed by ‘inflight WIFI service’ (0.145), ‘class’ (0.107), ‘type of Travel’ (0.096), ‘inflight entertainment’ (0.06), ‘seat comfort’ (0.049), and so on. Especially, there are many numbers of dissatisfied passengers who used economy or economy plus service in ‘class’ and their flights for personal travel purposes in ‘type of Travel’ compared to the other groups in the same parameter. Those particular inconveniences should be improved in the respective services. Thus, the airline companies in the US ought to provide better service in order to satisfy more passengers and not fall behind in the competition of attracting as many potential passengers as possible from now on by improving in order from the highest feature importance, online boarding service.

## License
[MIT](https://choosealicense.com/licenses/mit/)
