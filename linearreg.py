import numpy as np
import scipy.optimize as op
import csv
import math

# CostFunction gets a X that is  m x (n+1) for theta0
# y is m x 1 theta is (n+1) x 1
def costfunction(theta, X, y, lda ):
    m,n = X.shape
    theta = theta.reshape((n, 1))
    y = y.reshape((m, 1))
    error = np.dot(X, theta) - y
    term1 = (1/(2*m)) * sum(np.square(error))
    temp2 = np.vstack((np.zeros((1,1), dtype = np.int), theta[1:n]))
    term2 = (lda/(2*m))*sum(np.square(temp2))
    J = term1 + term2
    return J

def gradient(theta, X, y, lda ):
    m,n = X.shape
    theta = theta.reshape((n, 1))
    y = y.reshape((m, 1))
    error = np.dot(X, theta) - y
    term1 = (1/m) * np.dot((X.T), error)
    temp2 = np.vstack((np.zeros((1,1), dtype = np.int), theta[1:n]))
    term2 = (lda/m)*temp2
    grad = term1 + term2
    return grad.flatten()

def trainlinearregression ( X, y, lda, maxiter):
    m,n = X.shape
    print (str(m) + " " + str(n) )
    initial_theta = np.zeros((n, 1))
    result = op.minimize(fun = costfunction, x0 = initial_theta, args = (X, y, lda), method = 'TNC',
             jac = gradient, options ={ 'disp': False, 'maxiter': maxiter }  )
    optimal_theta = result.x
    return optimal_theta

def cost(theta, X, y):
    m,n = X.shape
    theta = theta.reshape((n, 1))
    y = y.reshape((m, 1))
    error = np.dot(X, theta) - y
    msqe = sum(np.square(error))/m
    return error,msqe

def nfldataread (pname) :
    with open('nflData.csv', 'r') as csvfile:
        nflreader = csv.reader(csvfile, delimiter=',')
        wfh = open ( (str(pname)+ ".csv"), 'w', newline="")
        wfha = csv.writer(wfh)
        count = 0
        countr = 0

        ymap = { '2010':1, '2011':2, '2012':3, '2013':4, '2014':5, '2015':6 }
        tmap = { 'BAL':1/32, 'CIN':2/32, 'CLE':3/32, 'PIT':4/32, 'CHI':5/32, 'DET':6/32, 'GB':7/32, 'MIN':8/32, 'HOU':9/32, 'IND':10,
                 'JAC':11/32, 'TEN':12/32, 'ATL':13/32, 'CAR':14/32, 'NO':15/32, 'TB':16/32, 'BUF':17/32, 'MIA':18/32, 'NE':19/32,
                 'NYJ':20/32, 'DAL':21/32, 'NYG':22/32, 'PHI':23/32, 'WAS':24/32, 'DEN':25/32, 'KC':26/32, 'OAK':27/32, 'SD':28/32,
                 'ARI':29/32, 'SF':30/32, 'SEA':31/32, 'STL':32/32 }
        for row in nflreader:

            for i in [1,6,11,14,21,27,30,32,34,35,36,40,44,45,51,56,59] :
                if row[i] == '' :
                    row[i] = 0
                    #Zeroing out empty params

            if count != 0 :
                year = str(row[0]) #i
                game_eid = row[1] #i
                home_team = row[2] #i
                away_team = row[3] #i
                #score_home = row[4]
                #score_away= row[5]
                fumbles_tot = int(row[6])
                rushing_yards = int(row[11]) #o
                #receiving_lngtd = row[14]
                #rushing_twopta = row[21] #i
                rushing_tds = int(row[27]) #o
                #receiving_rec = row[30]
                #receiving_twopta = row[32]
                receiving_yds = int(row[34])
                #rushing_att = row[35] #i
                reciving_twoptm = row[36] #o
                #rushing_lngtd = row[40]
                #receiving_lng = row[44]
                #pos = row[45]
                receiving_tds = int(row[51]) #o
                name =row[54]
                rushing_twoptm = row[56] #o
                #rushing_lng = row[59]
                team = row[61] #i

                if ((name == pname)) :
                    map_year = ymap[year] #m
                    if (team == home_team) :
                        playing_home =1 #m
                    else :
                        playing_home = 0 #m

                    if (team == home_team) :
                        played_against = tmap[away_team] #m
                    else :
                        played_against = tmap[home_team] #m

                    temp = str(game_eid)
                    month_played = int(temp[4:6]) #m

                    total_points = ((rushing_tds+ receiving_tds)*6) + (( rushing_yards +receiving_yds)/10) \
                                   -(fumbles_tot*2)

                    string = [ str(map_year), str(playing_home), str(played_against) ,str(month_played),
                       str(total_points) ]
                    wfha.writerow(string)
                    countr = countr +1
            count = count + 1
        wfh.close()
    #print (count) #Total records
    #print (countr) #Matched records



def main():

    lda = 0.01
    maxiter = 500
    rblist = ['A.Peterson', 'M.Lynch', 'F.Gore', 'C.Spiller', 'A.Foster', 'R.Jennings', 'C.Johnson',
              'D.McFadden','M.Forte', 'L.McCoy', 'J.Charles', 'F.Jackson' ]

    for player in rblist:
        nfldataread(player)
        Xtemp = np.loadtxt( (player +'.csv'), dtype = float, delimiter = ',', usecols = range(4) )
        mtr,ntr = np.shape(Xtemp)
        Xtrain = np.hstack ((np.ones ((mtr, 1)), Xtemp))
        Ytrain = np.loadtxt( (player + '.csv'), dtype = float, delimiter = ',', usecols = (4,) )
        theta = trainlinearregression( Xtrain, Ytrain, lda, maxiter)
        print(theta)
        errtr, msqetr = cost(theta, Xtrain, Ytrain)
        #print(errtr) # Per game Error
        print (player + "'s MSQE is:" + str(math.sqrt(msqetr)))


if __name__ == '__main__' :
    main()
else :
    print ("Didn't Work")