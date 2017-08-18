#!/usr/bin/python


def outlierCleaner(predictions, ages, net_worths):
    """
        Clean away the 10% of points that have the largest
        residual errors (difference between the prediction
        and the actual net worth).

        Return a list of tuples named cleaned_data where 
        each tuple is of the form (age, net_worth, error).
    """

    cleaned_data = []

    tmp = [(ages[idx], net_worths[idx], (predictions[idx]-net_worths[idx])**2) for
           idx in range(len(ages))]
    sorted_tmp = sorted(tmp, key=lambda student: student[2], reverse=True) # sort on error

    tenpercent = int(len(ages)*0.1)
    cleaned_data = sorted_tmp[tenpercent:]

    print sorted_tmp[:tenpercent]
    print len(ages)
    print len(cleaned_data)

    return cleaned_data

