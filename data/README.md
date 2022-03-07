# Data folder

This folder contained the data sets used in this work.

## MNIST and BMNIST

MNIST data set is a very popular collection of images, a complete description is available [here](http://yann.lecun.com/exdb/mnist/).
It is composed of a training set of 60,000 examples, and a test set of 10,000 examples.
These are gray level images of $28\times28$ pixels representing handwritten digits from 0 to 9, and thus composed by 10 classes.
The digits have been size-normalized and centered in a fixed-size image.
The sample population is uniform w.r.t. the classes.

In order to have a baseline data set that could be comparable to the EUROMDS binary data set, we chose to binarize those images.
Each image pixel is then scaled in the range $[0,1]$, since the values of pixels' gray levels are provided in the range $[0,255]$.
Then they are rounded to assume values 0 or 1 with the usual rounding threshold of 0.5.
The transformed data set will be then denoted as Binary MNIST (BMNIST).

![mnist_example](../images/mnist_example.png?raw=true)
![bmnist_example](../images/bmnist_example.png?raw=true)

## EUROMDS DATA SET

The data on Myelodisplastic Syndrome (MDS) that are actually available between the [GenoMed4ALL](https://genomed4all.eu/) consortium are a combination of genomic, cytogenetic and clinical data.
This data is not public available.
GenoMed4ALL is an EU-funded project that aims to accumulate evidence points towards personalised medicine to treat and manage common, rare and ultra-rare haematological diseases.
To address this need and support research further, the project is developing a data-sharing platform that utilises novel advanced AI models.
The project's partners are sharing infrastructure, powerful computing facilities, hospital registries and data processing tools towards this effort.
The project is contributing to the exploitation of omics and clinical data in patient-oriented research and decision-making through advanced statistics and machine learning approaches.
These tools are developed in the hope to obtain improved therapies and better clinical outcomes.

I will report here a brief description of EUROMDS data set.
The data set is in a "csv" format and we have that rows represent patients while the columns are the variables (features).
These variables are organized as follows:

### General and demographic variables

- Patient ID, EUROMDS patient labels
- Gender, Male or Female
- Age at data collection, age at diagnosis

### Clinical variables (prognostic scores)

- World Health Organization (WHO) 2016 subtype, WHO disease classification
-International Prognostic Scoring System (IPSS) risk group, disease classification according to IPSS
- Revised International Prognostic Scoring System (IPSSR) risk group, disease classification according to IPSSR

### Clinical-biological variables at diagnosis

- These variables are composed of haematochemical tests results on blood (cell counts and ferritin etc.) and bone marrow (count of bone marrow blasts an sideroblasts etc.) and a comorbidity score index.

### Follow up and outcome variables

- Leukemia free survival, characterized by two variable: event, and time to event
- Overall survival: event, and time to event
- Acute Myeloid Leukemia (AML) adjusted overall survival: event, and time to event

### Cytogenetic variables

- These data are grouped in 13 columns where is reported the presence/absence of a chromosomal alteration.
This data is represented by 0 when chromosomal alteration absent, 1 when chromosomal alteration is present, "NaN" if not measured.

### Genomic variables

- These variables are composed by the mutational status of 47 selected genes (gene panel for MDS).
This data is represented by 0 where mutation is absent, 1 where mutation is present, "NaN" if not measured.

### HDP components

- Patient specific Hierarchical Dirichlet Processes (HDP).
The weights across 6 latent components are listed as reported in \cite{bersanelli2021classification}.

In this work only a part of these variables were used: from the sets of genomic variables, and cytogenetic variables were extracted those variables for training our DEC model, precisely the same used in \cite{bersanelli2021classification} for HDP; follow up and outcome variables were used to characterize the clusters obtained by our DEC model; HDP components were used as "ground truth" to evaluate our DEC model clustering results.
Since the columns of variables used in the end were 54, we decided to give simple representation of a patient as an image $6\times9$ were every pixel corresponds to a specific variable as described by Tthe following table.
These kind of visual representations are often helpful for clinicians because they are able to recognize features with a look.

<table align="center">
    <tr>
        <td align="center"> ICHR17qort17p </td>
        <td align="center"> IDH2 </td>
        <td align="center"> ETV6 </td>
        <td align="center"> PIGA </td>
        <td align="center"> SF3B1 </td>
        <td align="center"> ATRX </td>
    </tr>
    <tr>
        <td align="center"> IDH1 </td>
        <td align="center"> NOTCH1 </td>
        <td align="center"> LOCHR13OD13q </td>
        <td align="center"> ASXL1 </td>
        <td align="center"> FLT3 </td>
        <td align="center"> BCOR </td>
    </tr>
    <tr>
        <td align="center"> LOCHR5OD5qPLUSother </td>
        <td align="center"> BCORL1 </td>
        <td align="center"> TET2 </td>
        <td align="center"> ZRSR2 </td>
        <td align="center"> DNMT3A </td>
        <td align="center"> LOCHR12OD12P12p </td>
    </tr>
    <tr>
        <td align="center"> RUNX1 </td>
        <td align="center"> SMC1A </td>
        <td align="center"> LOCHR20OD20q </td>
        <td align="center"> MPL </td>
        <td align="center"> BRAF </td>
        <td align="center"> idicXq13 </td>
    </tr>
    <tr>
        <td align="center"> NRAS </td>
        <td align="center"> NF1 </td>
        <td align="center"> SMC3 </td>
        <td align="center"> PHF6 </td>
        <td align="center"> LOCHR7OD7q </td>
        <td align="center"> LOCHR9OD9q </td>
    </tr>
    <tr>
        <td align="center"> EZH2 </td>
        <td align="center"> U2AF1 </td>
        <td align="center"> GNAS </td>
        <td align="center"> WT1 </td>
        <td align="center"> GNB1 </td>
        <td align="center"> RAD21 </td>
    </tr>
    <tr>
        <td align="center"> SRSF2 </td>
        <td align="center"> TP53 </td>
        <td align="center"> CBL </td>
        <td align="center"> KRAS </td>
        <td align="center"> FBXW7 </td>
        <td align="center"> del5q </td>
    </tr>
    <tr>
        <td align="center"> LOCHR11OD11q </td>
        <td align="center"> PTPN11 </td>
        <td align="center"> GATA2 </td>
        <td align="center"> KIT </td>
        <td align="center"> NPM1 </td>
        <td align="center"> GOCHR8 </td>
    </tr>
    <tr>
        <td align="center"> JAK2 </td>
        <td align="center"> LOCHRY </td>
        <td align="center"> PRPF40B </td>
        <td align="center"> CEBPA </td>
        <td align="center"> CBLB </td>
        <td align="center"> STAG2 </td>
    </tr>
</table>

An example of this representation is shown by the following image.

![recon_mds](../images/recons_MDS.png?raw=true)

In order to fill those values where the measure could not be accomplished, and thus represented by "NaN", we decided to replace "NaN" with the empirical probability of having value 1 in that column.
This empirical probability was estimated by the frequency of 1s w.r.t. to all the patients that have that column filled with 1 or 0, i.e. excluding those with "NaN".
