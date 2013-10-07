# TB Model # August, 2013 # Structure of Model: # Uninfected ->Latent -> Active -> Recovered/Dead # 4 compartments: "SEIR" Model with sensitive and resistant MDR and INH mono
###################################
###################################

print "Rif + INH molecular test high"

# Install packages
###################################
import numpy as np
import scipy as sp
import matplotlib as mpl
import matplotlib.pyplot as plt
import csv
from scipy.integrate import odeint
plt.ion()
from scipy.optimize import fsolve
from pylab import *

# Define parameters
###################################

###General parameters
tb_num = 19.    # number of TB states = 19
beta_decMDR = 0.774 #> from with equipop
beta_decINH = 0.98629 #> from with equipop
react = 0.0005 # yearly rate of reactivating from latent to active TB 
tb_mort = 0.15  # yearly rate of death from active TB; for smear-pos would 0.2, but assuming pre-diagnostic phase that is half as long 
bl_mort = 1./60. # life expectancy of 60 yrs
prim_prog = 0.14  # proportion of infections that progress rapidly to active in sensitive
lat_prot = 0.45 # immune protection against reinfection if latently infected
self_cure = 0.1  # out of active TB: rate of spontaneous recovery per year = mort., so CFR (Case fatality rate[not a rate!but %])= 68%
neverdx=0.15 # Assuming 15% of patients never present to diagnosis

###### Likelihood of being diagnosed
# Sensitvity for TB diagnosis
sens_smearcx_dx = 0.8 # sensitivity of diagnosis with current diagnostic standard taking into account repeat diagnostic attempts
sens_mol_dx = 0.95 # sensitivity of diagnosis with molecular diagnostic taking into account repeat diagnostic attempts 
 
# Sensitivity for diagnosis of resistance
sens_molRif = 0.94 # sensitivity of molecular test for Rif resistance > this ignores the fact that sensitivity will be lower in partially treated failure patients
sens_molINH = 0.88 # sensitivity of molecular test for high level INH resistance > this ignores the fact that sensitivity will be lower in partially treated failure patients

#Propability of different groups of patients getting certain tests
# All patients get a diagnostic test
prop_molRif_new = 0.5 # proportion of population that receives molecular testing to detect TB and Rif resistance (i.e. Xpert) when first presenting (RIF resistant treated as MDR)
prop_molRif_default = 0.8 # proportion of population that receives molecular testing to detect TB and Rif  resistance (i.e. Xpert) when defaulting (RIF resistant treated as MDR)
prop_molRif_fail =  1. # proportion of population that receives molecular testing to detect TB and Rif resistance (i.e. Xpert) when failing (RIF resistant treated as MDR)
prop_INHtest=1. #proportion who gets INH test out of the ones that get Xpert-like test

##### Rates
txprob=0.85 #treatment probability

dx_rate= 2.*txprob 
dx_rateF= 4.*txprob 
dx_rateDef= 2.*txprob 

##### NEW CASES 
# Proportions depend on whether patients get appropriately diagnosed with resistance or not and what the effectiveness of therapy is
# Patients who do not get diagnosed with TB remain in compartment; patients who do not get diagnosed with resistance get substandard therapy
# Model assumes that patients who do not get molecular resistance testing, do not get resistance testing at all

### Sensitive: Proportions that fail, cure, default with ACTIVE THERAPY in sensitive of those that don't die or self-cure
curedS=0.88
propdefS=0.75 
acq_resS=0.167
propINH_acq_resS=0.2

prop_cureS= curedS*((prop_molRif_new*sens_mol_dx)+(1-prop_molRif_new)*sens_smearcx_dx) #India WHO estimate
prop_failS=(1-propdefS)*(1-acq_resS)*(1-curedS)*(prop_molRif_new*sens_mol_dx+(1-prop_molRif_new)*sens_smearcx_dx) #SEA WHO estimate 0.02
prop_defaultS=propdefS*(1-curedS)*(prop_molRif_new*sens_mol_dx+(1-prop_molRif_new)*sens_smearcx_dx) #India WHO estimate, estimate of WHO: 5 defaulters;  4% relapse from Menzies 2009 PLOS Med
prop_INHS=(1-propdefS)*acq_resS*propINH_acq_resS*(1-curedS)*(prop_molRif_new*sens_mol_dx+(1-prop_molRif_new)*sens_smearcx_dx) #Lew Ant Int Med 2008
prop_MDRS=(1-propdefS)*acq_resS*(1-propINH_acq_resS)*(1-curedS)*(prop_molRif_new*sens_mol_dx+(1-prop_molRif_new)*sens_smearcx_dx) #Lew Ant Int Med 2008

### MDR: Proportions that fail, default, cure in MDR depending on resistance testing
# If diagnosed to have resistance gets active treatment, otherwise standard retreatment

# Proportion of fail/cure/default if treated based on DST
curedR_DST=0.52
propdefR_DST=0.542

prop_cureR_RxDST=curedR_DST # SEA WHO estimate
prop_failR_RxDST=(1-propdefR_DST)*(1-curedR_DST) #based on data from Franke CID 2013, Lee IJTLD 2011 > 0.20 fail, 0.22 recurrence
prop_defaultR_RxDST=propdefR_DST*(1-curedR_DST)

# Proportion of fail/cure/default if treated with standard therapy
curedR_Stand=0.25 
propdefR_Stand=0.467 

prop_cureR_RxStand=curedR_Stand #WHO report, Hong Kong MRC trial, Espinal JAMA 2000, Seung CID 2004
prop_failR_RxStand= (1-propdefR_Stand)*(1-curedR_Stand) # WHO report, Migliori IJTLD 2002
prop_defaultR_RxStand=propdefR_Stand*(1-curedR_Stand)

prop_failR=prop_molRif_new*sens_mol_dx*sens_molRif*prop_failR_RxDST + prop_molRif_new*sens_mol_dx*(1-sens_molRif)*prop_failR_RxStand + (1-prop_molRif_new)*sens_smearcx_dx*prop_failR_RxStand  # Prop of failure in diagnosed:Proportion that gets test*diagnostic sensitivity (sensitivity for TB detection and for resistance)*failure prop + Prop of failure who are diagnosed with TB but not with resistance)
prop_defaultR=prop_molRif_new*sens_mol_dx*sens_molRif*prop_defaultR_RxDST + prop_molRif_new*sens_mol_dx*(1-sens_molRif)*prop_defaultR_RxStand + (1-prop_molRif_new)*sens_smearcx_dx*prop_defaultR_RxStand
prop_cureR=prop_molRif_new*sens_mol_dx*sens_molRif*prop_cureR_RxDST + prop_molRif_new*sens_mol_dx*(1-sens_molRif)*prop_cureR_RxStand + (1-prop_molRif_new)*sens_smearcx_dx*prop_cureR_RxStand 


### in INH resistant
# Proportion of fail/cure/default if treated based on DST
curedI_DST=0.88
propdefI_DST=0.75
acq_resI_DST=0.033

prop_cureI_RxDST=curedI_DST 
prop_failI_RxDST=(1-propdefI_DST)*(1-acq_resI_DST)*(1-curedI_DST)
prop_defaultI_RxDST=propdefI_DST*(1-curedI_DST)
prop_MDRI_RxDST=(1-propdefI_DST)*acq_resI_DST*(1-curedI_DST)
#
# Proportion of fail/cure/default if treated with standard therapy
curedI_Stand=0.80
propdefI_Stand=0.45
acq_resI_Stand=0.09

prop_cureI_RxStand=curedI_Stand    #WHO report, Hong Kong MRC trial, Espinal JAMA 2000, Seung CID 2004
prop_failI_RxStand=(1-propdefI_Stand)*(1-acq_resI_Stand)*(1-curedI_Stand) # WHO report, Lew IJTLD 2011
prop_defaultI_RxStand=propdefI_Stand*(1-curedI_Stand)
prop_MDRI_RxStand=(1-propdefI_Stand)*acq_resI_Stand*(1-curedI_Stand) #Lew IJTLD 2011
#
prop_failI= prop_molRif_new*sens_mol_dx*prop_INHtest*sens_molINH*prop_failI_RxDST + prop_molRif_new*prop_INHtest*sens_mol_dx*(1-sens_molINH)*prop_failI_RxStand+ prop_molRif_new*sens_mol_dx*(1-prop_INHtest)*prop_failI_RxStand + (1-prop_molRif_new)*sens_smearcx_dx*prop_failI_RxStand
prop_defaultI= prop_molRif_new*sens_mol_dx*prop_INHtest*sens_molINH*prop_defaultI_RxDST + prop_molRif_new*prop_INHtest*sens_mol_dx*(1-sens_molINH)*prop_defaultI_RxStand+ prop_molRif_new*sens_mol_dx*(1-prop_INHtest)*prop_defaultI_RxStand + (1-prop_molRif_new)*sens_smearcx_dx*prop_defaultI_RxStand
prop_cureI=prop_molRif_new*sens_mol_dx*prop_INHtest*sens_molINH*prop_cureI_RxDST + prop_molRif_new*prop_INHtest*sens_mol_dx*(1-sens_molINH)*prop_cureI_RxStand+ prop_molRif_new*sens_mol_dx*(1-prop_INHtest)*prop_cureI_RxStand + (1-prop_molRif_new)*sens_smearcx_dx*prop_cureI_RxStand
prop_MDRresI = prop_molRif_new*sens_mol_dx*prop_INHtest*sens_molINH*prop_MDRI_RxDST + prop_molRif_new*prop_INHtest*sens_mol_dx*(1-sens_molINH)*prop_MDRI_RxStand+ prop_molRif_new*sens_mol_dx*(1-prop_INHtest)*prop_MDRI_RxStand + (1-prop_molRif_new)*sens_smearcx_dx*prop_MDRI_RxStand
#

##### Treatment outcomes for relapse/default/failure and reinfection

retreatdec=0.9 #retreated cases will get 10% lower proportion of cure

### Sensitive: Proportions that fail, cure, default with ACTIVE THERAPY in sensitive of those that don't die or self-cur
curedDef_S=0.88*retreatdec  
propdefDef_S=0.75 
acq_resDef_S=0.167
propINH_acq_resDef_S=0.2

propDef_cureS= curedDef_S*(prop_molRif_default*sens_mol_dx+(1-prop_molRif_default)*sens_smearcx_dx) #India WHO estimate
propDef_failS=(1-propdefDef_S)*(1-acq_resDef_S)*(1-curedDef_S)*(prop_molRif_default*sens_mol_dx+(1-prop_molRif_default)*sens_smearcx_dx) #SEA WHO estimate 0.02
propDef_defaultS=propdefDef_S*(1-curedDef_S)*(prop_molRif_default*sens_mol_dx+(1-prop_molRif_default)*sens_smearcx_dx) #India WHO estimate, estimate of WHO: 5 defaulters;  4% relapse from Menzies 2009 PLOS Med
propDef_INHS=(1-propdefDef_S)*acq_resDef_S*propINH_acq_resDef_S*(1-curedDef_S)*(prop_molRif_default*sens_mol_dx+(1-prop_molRif_default)*sens_smearcx_dx) #Lew Ant Int Med 2008
propDef_MDRS=(1-propdefDef_S)*acq_resDef_S*(1-propINH_acq_resDef_S)*(1-curedDef_S)*(prop_molRif_default*sens_mol_dx+(1-prop_molRif_default)*sens_smearcx_dx) #Lew Ant Int Med 2008

### MDR: Proportions that fail, default, cure in MDR depending on resistance testing
# If diagnosed to have resistance gets active treatment, otherwise standard retreatment

# Proportion of fail/cure/default if treated based on DST
curedRDef_DST=0.52*retreatdec 
propdefRDef_DST=0.542

propDef_cureR_RxDST=curedRDef_DST # SEA WHO estimate
propDef_failR_RxDST=(1-propdefRDef_DST)*(1-curedRDef_DST) #based on data from Franke CID 2013, Lee IJTLD 2011 > 0.20 fail, 0.22 recurrence
propDef_defaultR_RxDST=propdefRDef_DST*(1-curedRDef_DST)

# Proportion of fail/cure/default if treated with standard therapy
curedRDef_Stand=0.25 #*retreatdec 
propdefRDef_Stand=0.467 

propDef_cureR_RxStand=curedRDef_Stand #WHO report, Hong Kong MRC trial, Espinal JAMA 2000, Seung CID 2004
propDef_failR_RxStand= (1-propdefRDef_Stand)*(1-curedRDef_Stand) # WHO report, Migliori IJTLD 2002
propDef_defaultR_RxStand=propdefRDef_Stand*(1-curedRDef_Stand)

propDef_failR=prop_molRif_default*sens_mol_dx*sens_molRif*propDef_failR_RxDST + prop_molRif_default*sens_mol_dx*(1-sens_molRif)*propDef_failR_RxStand + (1-prop_molRif_default)*sens_smearcx_dx*propDef_failR_RxStand  # Prop of failure in diagnosed:Proportion that gets test*diagnostic sensitivity (sensitivity for TB detection and for resistance)*failure prop + Prop of failure who are diagnosed with TB but not with resistance)
propDef_defaultR=prop_molRif_default*sens_mol_dx*sens_molRif*propDef_defaultR_RxDST + prop_molRif_default*sens_mol_dx*(1-sens_molRif)*propDef_defaultR_RxStand + (1-prop_molRif_default)*sens_smearcx_dx*propDef_defaultR_RxStand
propDef_cureR=prop_molRif_default*sens_mol_dx*sens_molRif*propDef_cureR_RxDST + prop_molRif_default*sens_mol_dx*(1-sens_molRif)*propDef_cureR_RxStand + (1-prop_molRif_default)*sens_smearcx_dx*propDef_cureR_RxStand 


### in INH resistant
# Proportion of fail/cure/default if treated based on DST
curedIDef_DST=0.88*retreatdec
propdefIDef_DST=0.75
acq_resIDef_DST=0.033

propDef_cureI_RxDST=curedIDef_DST 
propDef_failI_RxDST=(1-propdefIDef_DST)*(1-acq_resIDef_DST)*(1-curedIDef_DST)
propDef_defaultI_RxDST=propdefIDef_DST*(1-curedIDef_DST)
propDef_MDRI_RxDST=(1-propdefIDef_DST)*acq_resIDef_DST*(1-curedIDef_DST)
#
# Proportion of fail/cure/default if treated with standard therapy
curedIDef_Stand=0.80*retreatdec
propdefIDef_Stand=0.45
acq_resIDef_Stand=0.09

propDef_cureI_RxStand=curedIDef_Stand    #WHO report, Hong Kong MRC trial, Espinal JAMA 2000, Seung CID 2004
propDef_failI_RxStand=(1-propdefI_Stand)*(1-acq_resIDef_Stand)*(1-curedIDef_Stand) # WHO report, Lew IJTLD 2011
propDef_defaultI_RxStand=propdefI_Stand*(1-curedIDef_Stand)
propDef_MDRI_RxStand=(1-propdefI_Stand)*acq_resIDef_Stand*(1-curedIDef_Stand) #Lew IJTLD 2011
#
propDef_failI= prop_molRif_default*sens_mol_dx*prop_INHtest*sens_molINH*propDef_failI_RxDST + prop_molRif_default*sens_mol_dx*prop_INHtest*(1-sens_molINH)*propDef_failI_RxStand + prop_molRif_default*sens_mol_dx*(1-prop_INHtest)*propDef_failI_RxStand +(1-prop_molRif_default)*sens_smearcx_dx*propDef_failI_RxStand  #Prob of failure if diagnosed Prop of failure in diagnosed to have TB but not INH resistance 
propDef_defaultI= prop_molRif_default*sens_mol_dx*prop_INHtest*sens_molINH*propDef_defaultI_RxDST + prop_molRif_default*sens_mol_dx*prop_INHtest*(1-sens_molINH)*propDef_defaultI_RxStand + prop_molRif_default*sens_mol_dx*(1-prop_INHtest)*propDef_defaultI_RxStand +(1-prop_molRif_default)*sens_smearcx_dx*propDef_defaultI_RxStand 
propDef_cureI= prop_molRif_default*sens_mol_dx*prop_INHtest*sens_molINH*propDef_cureI_RxDST + prop_molRif_default*sens_mol_dx*prop_INHtest*(1-sens_molINH)*propDef_cureI_RxStand + prop_molRif_default*sens_mol_dx*(1-prop_INHtest)*propDef_cureI_RxStand + (1-prop_molRif_default)*sens_smearcx_dx*propDef_cureI_RxStand 
propDef_MDRresI = prop_molRif_default*sens_mol_dx*prop_INHtest*sens_molINH*propDef_MDRI_RxDST + prop_molRif_default*sens_mol_dx*prop_INHtest*(1-sens_molINH)*propDef_MDRI_RxStand + prop_molRif_default*sens_mol_dx*(1-prop_INHtest)*propDef_MDRI_RxStand + (1-prop_molRif_default)*sens_smearcx_dx*propDef_MDRI_RxStand 
#


# Define compartments:
###################################

# U: Uninfected [0]
# Ls: Latent sensitive [1]
# As: Active sensitive [2]
# Fs: Failed sensitive [3]
# RLs: Relapse sensitive [4]
# Ns: Active TB sensitive never diagnosed [5]
# Lr: Latent INH/RIF resistant [6]
# Ar Active INH/RIF resistant [7]
# Fr: Failed INH/RIF resistant [8]
# RLr: Relapse INH/RIF resistant [9]
# Nr: Active TB MDR never diagnosed [17]
# Li: Latent INH resistant [11]
# Ai Active INH resistant [12]
# Fi: Failed INH resistant [13]
# Ni: Active TB INH never diagnosed [15]
# RLi: Relapse INH resistant [14]
# Rs: Recovered sensitive [16]
# Rr: Recovered INH/RIF resistant [17]
# Ri: Recovered INH resistant [18]

# Define transitions
########################

# U->Ls: infection, minus primary progression in sensitive
# U->Lr: infection, minus primary progression in MDR 
# U->Li: infection, minus primary progression in INH resistant

# U->As: infection with primary progression in sensitive
# U->Ar: infection with primary progression in MDR resistant
# U->Ai: infection with primary progression in INH resistant

# Lr->Ar: reactivation + reinfection after self-cure in MDR (reinfection decreased due to immunity from LTBI)
# Li->Ai: reactivation + reinfection after self-cure in INH resistant (reinfection decreased due to immunity from LTBI)
# Ls->As: reactivation + reinfection after self-cure in sensitive (reinfection decreased due to immunity from LTBI)

# As->Cs: treatment in sensitive cured (with sufficient treatment independent of default/ treatment completed)
# Ar->Cr: treatment in MDR cured (with sufficient treatment independent of default/ treatment completed)
# Ai->Ci: treatment in INH resistant cured (with sufficient treatment independent of default/ treatment completed)

# Ai->Ri: defaulters that will relapse INH resistant
# As->Rs: defaulters that will relapse sensitive
# Ar->Rr: defaulters that will relapse MDR

# As->Fs: failing sensitive
# Ar->Fr: failing MDR resistant
# Ai->Fi: failing INH resistant
# Ai->Fr: failing INH resistant developing MDR 
# >>> no mortality as patients who die do not make it out of active box
# >>> patients in failing group are infectious but at a discounted rate (comparable to smear negative 1/5), no increased death rate

# Fi->Ri: failures INH resistant that will get cured (default or treatment completed) and then relapse 
# Fr->Rr: failures MDR that will get cured (default or treatment completed) and then relapse  
# Fs->Rs: failures sensitive that will get cured (default or treatment completed) and then relapse 

# Fs->Fr: failures sensitive that develop MDR
# Fr->Fi: failures sensitive that develop INH resistance
# Fi->Fr: failures INH resistant that develop MDR

# Cs->Rs: relapse/reinfection from cured/defaulted in sensitive 
# Cr->Rr: relapse/reinfection from cured/defaulted in MDR 
# Ci->Ri: relapse/reinfection from cured/defaulted in INH resistant

# Rs->Fs: failing with retreatment sensitive
# Rr->Fr: failing with retreatment in MDR 
# Ri->Fi: failing with retreatment in INH resistant
# Rs->Fi: failing with retreatment sensitive developing INH resistance
# Rs->Fr: failing with retreatment sensitive developing MDR 
# Ri->Fr: failing with retreatment INH monoresistant developing MDR

# >> reinfection possible out of every recovered and latent compartment considering protective effect of prior infection 
# >> exit from each compartment = mortality
# >> exit from each active department: self-cure
# >> to maintain constant population, make exit = entry 
# >> assume entries are uninfected


# Routine to define mortality
###################################

# active and retreatment groups have increased mortality
def mortality(comp):
    mu = bl_mort
    if comp == 2:
        mu += tb_mort
    if comp == 7:
        mu += tb_mort
    if comp == 12:
        mu += tb_mort
    if comp == 3:
        mu += tb_mort*0.25
    if comp == 8:
        mu += tb_mort*0.25
    if comp == 13:
        mu += tb_mort*0.25
    if comp == 4:
        mu += tb_mort
    if comp == 9:
        mu += tb_mort
    if comp == 14:
        mu += tb_mort
    if comp == 5:
        mu += tb_mort
    if comp == 10:
        mu += tb_mort
    if comp == 15:
        mu += tb_mort
    return mu

# Define initial conditions:
###################################
#retrieve pop 2010 from 2010 with equipop
pop2010=np.load("Dropbox/Montreal/Projects/Modeling impact of scale-up/SEIR INH monoresistance/code\\pop2010.npy")
beta=pop2010[19]
popn=np.zeros(19)
popn[0:19] = pop2010[0:19]
print popn
sumpop=sum(popn)
#
# First calculate beta to achieve decrease in incidence of 2-3% over time
# then with revised beta dial in molecular test
beta=6.77464442344# revised from calculator of decreased in incidence


# Differential equation function:
###################################

def change(pop,t):
    P = pop
    dxdt = np.zeros(tb_num)
    mortv = np.zeros(tb_num)
    for i in range(19):                  # calculate sum of all deaths
        mortv[i]=mortality(i)*P[i]
    mort_total = sum(mortv)
    forceS = beta*(P[2]+P[4]+P[5]+0.2*P[3])/100000 # infectious patients are in active/never diagnosed groups and reduced infectiousness failure group (partial treatment)
    forceMDR = beta*beta_decMDR*(P[7]+P[9]+P[10]+0.2*P[8])/100000
    forceINH = beta*beta_decINH*(P[12]+P[14]+P[15]+0.2*P[13])/100000
    
    # define equations - cannot use loops
    dxdt[0] = mort_total \
              -forceS*P[0] \
              -forceMDR*P[0] \
              -forceINH*P[0] \
              -mortality(0)*P[0]
    
    # sensitive LTBI group
    dxdt[1] = forceS*P[0]*(1-prim_prog) \
              + forceS*P[6]*(1-lat_prot)*(1-prim_prog) \
              + forceS*P[11]*(1-lat_prot)*(1-prim_prog) \
              + self_cure*P[2] \
              + self_cure*P[5] \
              - forceS*P[1]*(1-lat_prot)*prim_prog \
              - forceMDR*P[1]*(1-lat_prot)*(1-prim_prog) \
              - forceINH*P[1]*(1-lat_prot)*(1-prim_prog) \
              - forceMDR*P[1]*(1-lat_prot)*prim_prog \
              - forceINH*P[1]*(1-lat_prot)*prim_prog \
              - react*P[1] \
              - mortality(1)*P[1]

    # sensitive active group new diagnosis
    dxdt[2] = forceS*P[0]*prim_prog*(1-neverdx) \
              +forceS*P[1]*(1-lat_prot)*prim_prog*(1-neverdx) \
              +forceS*P[6]*(1-lat_prot)*prim_prog*(1-neverdx) \
              +forceS*P[11]*(1-lat_prot)*prim_prog*(1-neverdx) \
              +react*P[1]*(1-neverdx)  \
              -dx_rate*prop_failS*P[2] \
              -dx_rate*prop_defaultS*P[2] \
              -dx_rate*prop_cureS*P[2] \
              -dx_rate*prop_INHS*P[2] \
              -dx_rate*prop_MDRS*P[2] \
              -self_cure*P[2] \
              -mortality(2)*P[2]

    # sensitive failed (patients that fail stay in failure unless they develop resistance)                 
    dxdt[3] =dx_rate*prop_failS*P[2] \
              +dx_rateDef*propDef_failS*P[4] \
              -dx_rateF*P[3]*propDef_INHS \
              -dx_rateF*P[3]*propDef_MDRS \
              -dx_rateF*propDef_defaultS*P[3] \
              -dx_rateF*propDef_cureS*P[3]\
              -mortality(3)*P[3]
    
    # sensitive retreatment              
    dxdt[4] =dx_rate*prop_defaultS*P[2] \
              +dx_rateF*propDef_defaultS*P[3] \
              +forceS*P[18]*(1-lat_prot)*prim_prog \
              +forceS*P[17]*(1-lat_prot)*prim_prog \
              +forceS*P[16]*(1-lat_prot)*prim_prog \
              -dx_rateDef*propDef_cureS*P[4] \
              -dx_rateDef*propDef_INHS*P[4] \
              -dx_rateDef*propDef_MDRS*P[4] \
              -dx_rateDef*propDef_failS*P[4] \
              -self_cure*P[4] \
              -mortality(4)*P[4] 

    # sensitive active never diagnosed
    dxdt[5] = react*P[1]*neverdx\
               +forceS*P[0]*prim_prog*neverdx\
               +forceS*P[1]*(1-lat_prot)*prim_prog*neverdx \
               +forceS*P[6]*(1-lat_prot)*prim_prog*neverdx  \
               +forceS*P[11]*(1-lat_prot)*prim_prog*neverdx  \
               -self_cure*P[5] \
               -mortality(5)*P[5]
    
    # sensitive cured              
    dxdt[16] =dx_rate*prop_cureS*P[2] \
              +dx_rateF*propDef_cureS*P[3] \
              +dx_rateDef*propDef_cureS*P[4] \
              +forceS*P[17]*(1-lat_prot)*(1-prim_prog) \
              +forceS*P[18]*(1-lat_prot)*(1-prim_prog) \
              +self_cure*P[4] \
              -forceS*P[16]*(1-lat_prot)*prim_prog \
              -forceMDR*P[16]*(1-lat_prot)*prim_prog \
              -forceINH*P[16]*(1-lat_prot)*prim_prog \
              -forceMDR*P[16]*(1-lat_prot)*(1-prim_prog) \
              -forceINH*P[16]*(1-lat_prot)*(1-prim_prog) \
              -mortality(16)*P[16]

    # resistant MDR LTBI    
    dxdt[6] = forceMDR*P[0]*(1-prim_prog) \
              + forceMDR*P[1]*(1-lat_prot)*(1-prim_prog)\
              + forceMDR*P[11]*(1-lat_prot)*(1-prim_prog) \
              + self_cure*P[7] \
              + self_cure*P[10] \
              - forceMDR*P[6]*(1-lat_prot)*prim_prog \
              - forceS*P[6]*(1-lat_prot)*(1-prim_prog) \
              - forceINH*P[6]*(1-lat_prot)*(1-prim_prog) \
              - forceS*P[6]*(1-lat_prot)*prim_prog \
              - forceINH*P[6]*(1-lat_prot)*prim_prog \
              - react*P[6] \
              - mortality(6)*P[6]
            
    # resistant MDR active TB            
    dxdt[7] = forceMDR*P[0]*prim_prog*(1-neverdx) \
              + forceMDR*P[1]*(1-lat_prot)*prim_prog*(1-neverdx) \
              + forceMDR*P[6]*(1-lat_prot)*prim_prog *(1-neverdx)\
              + forceMDR*P[11]*(1-lat_prot)*prim_prog*(1-neverdx) \
              + react*P[6]*(1-neverdx) \
              - dx_rate*prop_failR*P[7] \
              - dx_rate*prop_defaultR*P[7] \
              - dx_rate*prop_cureR*P[7] \
              - self_cure*P[7] \
              - mortality(7)*P[7]
            
    # resistant MDR failed (patients that fail stay in failure unless they develop resistance)             
    dxdt[8] = dx_rate*prop_failR*P[7] \
              +dx_rate*prop_MDRresI*P[12] \
              +dx_rateF*propDef_MDRresI*P[13] \
              +dx_rateF*propDef_MDRS*P[3]\
              +dx_rateDef*propDef_MDRS*P[4] \
              +dx_rateDef*propDef_MDRresI*P[14] \
              +dx_rateDef*propDef_failR*P[9] \
              +dx_rate*prop_MDRS*P[2] \
              -dx_rateF*propDef_defaultR_RxDST*P[8] \
              -dx_rateF*propDef_cureR_RxDST*P[8] \
              -mortality(8)*P[8]
    
    # resistant MDR retreatment              
    dxdt[9] = dx_rate*prop_defaultR*P[7] \
              +dx_rateF*propDef_defaultR_RxDST*P[8] \
              +forceMDR*P[16]*(1-lat_prot)*prim_prog \
              +forceMDR*P[17]*(1-lat_prot)*prim_prog \
              +forceMDR*P[18]*(1-lat_prot)*prim_prog \
              -dx_rateDef*propDef_cureR*P[9] \
              -dx_rateDef*propDef_failR*P[9] \
              -self_cure*P[9] \
              -mortality(9)*P[9]

    # Resistant MDR active never diagnosed
    dxdt[10] = forceMDR*P[0]*prim_prog*neverdx \
               + forceMDR*P[1]*(1-lat_prot)*prim_prog*neverdx \
               + forceMDR*P[6]*(1-lat_prot)*prim_prog*neverdx  \
               + forceMDR*P[11]*(1-lat_prot)*prim_prog*neverdx  \
               + react*P[6]*neverdx \
               - self_cure*P[10] \
               - mortality(10)*P[10]
               
    # Resistant MDR cured              
    dxdt[17] = dx_rate*prop_cureR*P[7] \
               +dx_rateF*propDef_cureR_RxDST*P[8] \
               +dx_rateDef*propDef_cureR*P[9] \
               + forceMDR*P[16]*(1-lat_prot)*(1-prim_prog)\
               + forceMDR*P[18]*(1-lat_prot)*(1-prim_prog) \
               + self_cure*P[9] \
               - forceMDR*P[17]*(1-lat_prot)*prim_prog \
               - forceS*P[17]*(1-lat_prot)*prim_prog \
               - forceINH*P[17]*(1-lat_prot)*prim_prog \
               - forceS*P[17]*(1-lat_prot)*(1-prim_prog) \
               - forceINH*P[17]*(1-lat_prot)*(1-prim_prog) \
               - mortality(17)*P[17]

    # resistant INH LTBI
    dxdt[11] = forceINH*P[0]*(1-prim_prog) \
               + forceINH*P[1]*(1-lat_prot)*(1-prim_prog)\
               + forceINH*P[6]*(1-lat_prot)*(1-prim_prog) \
               + self_cure*P[12] \
               + self_cure*P[15] \
               - forceINH*P[11]*(1-lat_prot)*prim_prog \
               - forceS*P[11]*(1-lat_prot)*(1-prim_prog) \
               - forceMDR*P[11]*(1-lat_prot)*(1-prim_prog) \
               - forceS*P[11]*(1-lat_prot)*prim_prog \
               - forceMDR*P[11]*(1-lat_prot)*prim_prog \
               - react*P[11] \
               - mortality(11)*P[11]
            
    # resistant INH active TB
    dxdt[12] = forceINH*P[0]*prim_prog*(1-neverdx) \
               + forceINH*P[1]*(1-lat_prot)*prim_prog*(1-neverdx) \
               + forceINH*P[6]*(1-lat_prot)*prim_prog*(1-neverdx) \
               + forceINH*P[11]*(1-lat_prot)*prim_prog*(1-neverdx) \
               + react*P[11]*(1-neverdx) \
               - dx_rate*prop_failI*P[12] \
               - dx_rate*prop_defaultI*P[12] \
               - dx_rate*prop_cureI*P[12] \
               - dx_rate*prop_MDRresI*P[12] \
               - self_cure*P[12] \
               - mortality(12)*P[12]

    # resistant INH failed (failures stay in failure)              
    dxdt[13] = dx_rate*prop_failI*P[12] \
               + dx_rateF*propDef_INHS*P[3]\
               + dx_rateDef*propDef_INHS*P[4] \
               + dx_rateDef*propDef_failI*P[14] \
               + dx_rate*prop_INHS*P[2] \
               - dx_rateF*propDef_MDRresI*P[13] \
               - dx_rateF*propDef_defaultI*P[13] \
               - dx_rateF*propDef_cureI*P[13] \
               - mortality(13)*P[13] 
    
    # resistant INH retreatment             
    dxdt[14] = dx_rate*prop_defaultI*P[12] \
               +dx_rateF*propDef_defaultI*P[13] \
               +forceINH*P[16]*(1-lat_prot)*prim_prog \
               +forceINH*P[17]*(1-lat_prot)*prim_prog \
               +forceINH*P[18]*(1-lat_prot)*prim_prog \
               -dx_rateDef*propDef_cureI*P[14] \
               -dx_rateDef*propDef_MDRresI*P[14] \
               -dx_rateDef*propDef_failI*P[14] \
               -self_cure*P[14] \
               -mortality(14)*P[14]
            
    # Resistant INH active never diagnosed
    dxdt[15] = forceINH*P[0]*prim_prog*neverdx \
               + forceINH*P[1]*(1-lat_prot)*prim_prog*neverdx \
               + forceINH*P[6]*(1-lat_prot)*prim_prog*neverdx  \
               + forceINH*P[11]*(1-lat_prot)*prim_prog*neverdx  \
               + react*P[11]*neverdx \
               -self_cure*P[15] \
               -mortality(15)*P[15]

    # resistant INH cured
    dxdt [18] = dx_rate*prop_cureI*P[12] \
                + dx_rateF*propDef_cureI*P[13] \
                + dx_rateDef*propDef_cureI*P[14] \
                + forceINH*P[17]*(1-lat_prot)*(1-prim_prog) \
                + forceINH*P[16]*(1-lat_prot)*(1-prim_prog)\
                + self_cure*P[14] \
                - forceINH*P[18]*(1-lat_prot)*prim_prog \
                - forceS*P[18]*(1-lat_prot)*prim_prog \
                - forceMDR*P[18]*(1-lat_prot)*prim_prog \
                - forceS*P[18]*(1-lat_prot)*(1-prim_prog) \
                - forceMDR*P[18]*(1-lat_prot)* (1-prim_prog) \
                - mortality(18)*P[18]
    return dxdt

# Differential equation function:
###################################

duration = 10.# 10 year time period
timestep = 0.1 # calculate equations every 0.1 yrs (100 timesteps)
time_range = np.arange(0, duration+timestep, timestep) # vector for calculations

# Solve differential equation
###################################
futpop = np.zeros(101)
futpop = odeint(change,popn,time_range) # provides output in csv file

print futpop[100,:]
endpop = sum(futpop[100,:])
print endpop

# calculate the ODE's: function, initial pop, time
# creates an array of population at each timestep:
# starting in 2010
# each subsequent column is the population at next timestep


# CALCULATE INCIDENCE, PREVALENCE, MORTALITY, CDR
###################################
def incprevmort(pop):
    P = pop
    forceS = beta*(P[2]+P[4]+P[5]+0.2*P[3])/100000 # infectious patients are in active/never diagnosed groups and reduced infectiousness failure group (partial treatment)
    forceMDR = beta*beta_decMDR*(P[7]+P[9]+P[10]+0.2*P[8])/100000
    forceINH = beta*beta_decINH*(P[12]+P[14]+P[15]+0.2*P[13])/100000
    prev_total = 0.                      # Calculation for new cases
    mort_total = 0.
    inc_totalN = forceS*P[0]*prim_prog \
                + forceS*(1-lat_prot)*prim_prog*(P[1]+ P[6]+ P[11]) \
                + react*P[1] \
                + forceS*(1-lat_prot)*prim_prog*(P[18]+P[17]+P[16]) \
                + forceMDR*P[0]*prim_prog \
                + forceMDR*(1-lat_prot)*prim_prog*(P[1]+ P[6]+ P[11]) \
                + forceMDR*(1-lat_prot)*prim_prog*(P[18]+P[17]+P[16]) \
                + react*P[6] \
                + forceINH*P[0]*prim_prog\
                + forceINH*(1-lat_prot)*prim_prog*(P[1]+ P[6]+ P[11]) \
                + react*P[11] \
                + forceINH*(1-lat_prot)*prim_prog*(P[18]+P[17]+P[16]) 
    inc_totalre = dx_rate*prop_defaultS*P[2] \
                  + dx_rate*prop_failS*P[2] \
                  + dx_rateF*propDef_defaultS*P[3] \
                  + dx_rate*prop_defaultR*P[7] \
                  + dx_rate*prop_failR*P[7] \
                  + dx_rateF*propDef_defaultR_RxDST*P[8] \
                  + dx_rate*prop_defaultI*P[12] \
                  + dx_rate*prop_failI*P[12] \
                  + dx_rateF*propDef_defaultI*P[13]
    inc_MDRtruenew = forceMDR*P[0]*prim_prog \
                     + forceMDR*(1-lat_prot)*prim_prog*(P[1]+ P[6]+ P[11]) \
                     + react*P[6]                 
    #inc_totalN includes patients with new infection and reinfection after cure
    #inc_totalre includes default, relapse and failure cases
    for i in (2,7,12,3,8,13,4,9,14,5,10,15):
        prev_total += P[i]                # calculates TB prevalence including infection, reinfection, relapse, default and failure 
        mort_total += mortality(i)*P[i]   # TB mortality

    prevMDR = P[7]+P[8]+P[9]+P[10]
    durMDR = prevMDR/inc_MDRtruenew
    
    detecttotalN = dx_rate*(P[2]+P[7]+P[12])*(prop_molRif_new*sens_mol_dx+(1-prop_molRif_new)*sens_smearcx_dx) # new cases detected (not incident cases as excludes reinfection/relapse)
    detecttotalre= dx_rateDef*(P[4]+P[9]+P[14])*(prop_molRif_default*sens_mol_dx+(1-prop_molRif_default)*sens_smearcx_dx) + dx_rateF*(P[3]+P[8]+P[13])- dx_rateF*(P[8]*propDef_failR_RxDST+P[13]*propDef_failI+P[3]*propDef_failS)# cases detected after default, relapse or reinfection, failure
    detecttotalpropfail= (dx_rateF*(P[3]+P[8]+P[13])- dx_rateF*(P[8]*propDef_failR_RxDST+P[13]*propDef_failI+P[3]*propDef_failS))/(dx_rateDef*(P[4]+P[9]+P[14])*(prop_molRif_default*sens_mol_dx+(1-prop_molRif_default)*sens_smearcx_dx) + dx_rateF*(P[3]+P[8]+P[13])- dx_rateF*(P[8]*propDef_failR_RxDST+P[13]*propDef_failI+P[3]*propDef_failS))
    detecttot=detecttotalN+detecttotalre # all cases detected

    detectSN = dx_rate*P[2]*(prop_molRif_new*sens_mol_dx+(1-prop_molRif_new)*sens_smearcx_dx) 
    detectSre = dx_rateDef*P[4]*(prop_molRif_default*sens_mol_dx+(1-prop_molRif_default)*sens_smearcx_dx)+ dx_rateF*P[3]*(1-propDef_failS)
    detectSpropfail = (dx_rateF*P[3]*(1-propDef_failS))/(dx_rateDef*P[4]*(prop_molRif_default*sens_mol_dx+(1-prop_molRif_default)*sens_smearcx_dx)+ dx_rateF*P[3]*(1-propDef_failS)) 
    detectStot = detectSN + detectSre
    
    detectMDRN = dx_rate*P[7]*(prop_molRif_new*sens_mol_dx+(1-prop_molRif_new)*sens_smearcx_dx) 
    detectMDRre = dx_rateDef*P[9]*(prop_molRif_default*sens_mol_dx+(1-prop_molRif_default)*sens_smearcx_dx)+ dx_rateF*P[8]*(1-propDef_failR_RxDST)
    detectMDRpropfail = (dx_rateF*P[8]*(1-propDef_failR_RxDST))/(dx_rateDef*P[9]*(prop_molRif_default*sens_mol_dx+(1-prop_molRif_default)*sens_smearcx_dx)+ dx_rateF*P[8]*(1-propDef_failR_RxDST)) 
    detectMDRtot = detectMDRN + detectMDRre

    cdr_total = detecttot/(inc_totalN+inc_totalre)*100 # case detection RATIO (%) 
    
    detectINHN = dx_rate*P[12]*(prop_molRif_new*sens_mol_dx+(1-prop_molRif_new)*sens_smearcx_dx)
    detectINHre = dx_rateDef*P[14]*(prop_molRif_default*sens_mol_dx+(1-prop_molRif_default)*sens_smearcx_dx) + dx_rateF*P[13]*(1-propDef_failI)
    detectINHpropfail = (dx_rateF*P[13]*(1-propDef_failI))/(dx_rateDef*P[14]*(prop_molRif_default*sens_mol_dx+(1-prop_molRif_default)*sens_smearcx_dx) + dx_rateF*P[13]*(1-propDef_failI))
    detectINHtot = detectINHN + detectINHre

    curedMDR=dx_rateF*propDef_cureR*P[8]
    
    return inc_totalN, inc_totalre, detecttot, detecttotalN, detecttotalre, detecttotalpropfail, detectStot, detectSN, detectSre, detectSpropfail, detectMDRtot, detectMDRN, detectMDRre, detectMDRpropfail, detectINHtot, detectINHN, detectINHre, detectINHpropfail, cdr_total, prev_total, mort_total, curedMDR, durMDR 

inc_bl_total = np.zeros((101,23))
for x in range(101):
    inc_bl_total[x,:] = incprevmort(futpop[x,:])

#Proportion of MDR detected
propMDRN= np.zeros(101)
propMDRre= np.zeros(101)
propMDRtot= np.zeros(101)

propINHN= np.zeros(101)
propINHre= np.zeros(101)
propINHtot= np.zeros(101)

propMDRN[:]= inc_bl_total[:,11]/inc_bl_total[:,3] #among all new cases
propMDRre[:] =  inc_bl_total[:,12]/inc_bl_total[:,4] #among all retreatment cases
propMDRtot[:]=  inc_bl_total[:,10]/inc_bl_total[:,2]

propINHN[:]=  inc_bl_total[:,15]/inc_bl_total[:,3]
propINHre[:] =  inc_bl_total[:,16]/inc_bl_total[:,4]
propINHtot[:]=  inc_bl_total[:,14]/inc_bl_total[:,2]


print "total incidence", (inc_bl_total[100:,0] + inc_bl_total[100:,1])    
print "total incidence of new cases", inc_bl_total[100:,0]
print "total incidence of retreatment cases", inc_bl_total[100:,1]
print "CDR", inc_bl_total[100:,18]
#
print "Proportion failure among all retreatment cases", inc_bl_total[100:,5]
#
print "proportion of S among total cases detected", (inc_bl_total[100:,6]/inc_bl_total[100:,2])
print "proportion of S among new cases detected", (inc_bl_total[100:,7]/inc_bl_total[100:,3])
print "proportion of S among retreatment cases detected", (inc_bl_total[100:,8]/inc_bl_total[100:,4])
print "Proportion failure among S retreatment cases", inc_bl_total[100:,9]
#
print "proportion of MDR among total cases detected", propMDRtot[100:]
print "proportion of MDR among new cases detected", propMDRN[100:]
print "proportion of MDR among retreatment cases detected", propMDRre[100:]
print "Proportion failure among MDR retreatment cases", inc_bl_total[100:,13]
print "Cured MDR", inc_bl_total[100:,21]
print "Duration MDR", inc_bl_total[100:,22]
#
print "proportion of INH among total cases detected", propINHtot[100:]
print "proportion of INH among new cases detected", propINHN[100:]
print "proportion of INH among retreatment cases detected", propINHre[100:]
print "Proportion failure among INHr retreatment cases", inc_bl_total[100:,17]

# Excel file y-label
##inc_totalN
##inc_totalre
##prev_total
##mort_total
##cdr_total
##detect_tot
##detectINH_tot
##detectMDR_tot
##cured MDR failure
##duration MDR 
##propMDRN
##propMDRre
##propMDRtot
##propINHN
##propINHre
##propINHtot

# create CSV file:
##################################
with open("Dropbox/Montreal/Projects/Modeling impact of scale-up/SEIR INH monoresistance/Output/Rif_INH_mol_test_high_080413.csv",'wb') as f:
    writer=csv.writer(f)
    writer.writerow(futpop[:,0])
    writer.writerow(futpop[:,1])
    writer.writerow(futpop[:,2])
    writer.writerow(futpop[:,3])
    writer.writerow(futpop[:,4])
    writer.writerow(futpop[:,5])
    writer.writerow(futpop[:,6])
    writer.writerow(futpop[:,7])
    writer.writerow(futpop[:,8])
    writer.writerow(futpop[:,9])
    writer.writerow(futpop[:,10])
    writer.writerow(futpop[:,11])
    writer.writerow(futpop[:,12])
    writer.writerow(futpop[:,13])
    writer.writerow(futpop[:,14])
    writer.writerow(futpop[:,15])
    writer.writerow(futpop[:,16])
    writer.writerow(futpop[:,17])
    writer.writerow(futpop[:,18])
    writer.writerow(inc_bl_total[:,0]) # inc new
    writer.writerow(inc_bl_total[:,1]) # inc re
    writer.writerow(inc_bl_total[:,19]) # prev
    writer.writerow(inc_bl_total[:,20]) # mort
    writer.writerow(inc_bl_total[:,18]) # CDR
    writer.writerow(inc_bl_total[:,2]) # detect total
    writer.writerow(inc_bl_total[:,14]) # detect INH total
    writer.writerow(inc_bl_total[:,10]) # detect MDR total
    writer.writerow(inc_bl_total[:,21]) # cured MDR failure
    writer.writerow (inc_bl_total[:,22]) # Duration MDR Prevalence (all)/Incidence (true new)
    writer.writerow(propMDRN[:])
    writer.writerow(propMDRre[:])
    writer.writerow(propMDRtot[:])
    writer.writerow(propINHN[:])
    writer.writerow(propINHre[:])
    writer.writerow(propINHtot[:])
    

