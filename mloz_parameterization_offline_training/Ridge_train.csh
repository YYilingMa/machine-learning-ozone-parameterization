#!/bin/csh
# c-shell script for ozone parameterization based on ridge regression

# set env var
setenv basedir                    '/home/hk-project-iconart2/ou4895/UKESM'
setenv outputdir                  '/hkfs/work/workspace/scratch/ou4895-ukesm/output/RidgeModel_73yrsTrain_under50km_piCTRL_76lev/'
setenv train_yrs                  73
setenv op_train                   'on'
setenv op_test                    'off' # test the estimator

if ("$op_train" == 'on') then

mkdir -p $outputdir
cd $outputdir

# loop over longitudes
# foreach lon_batch (0)
@ lon_batch = 21
while ( $lon_batch <= 40 ) # lon_batch=0-47 

#echo "=================== Training on lon=$lon_i ==================="

sed "s/LON_BATCH/$lon_batch/g"            $basedir/Ridge_train_piCTRL.sample > tmp1
sed "s/NYEAR/$train_yrs/g"                tmp1 > Ridge_train_piCTRL_lon_batch$lon_batch.py

cat > job_ridge_train_piCTRL << ENDFILE
#!/bin/bash -x
#SBATCH --nodes=1
#SBATCH --time=06:00:00
#SBATCH --partition=large
#SBATCH --mem=501600mb
#SBATCH --ntasks-per-node=76
#SBATCH --mail-type="END"
#SBATCH --mail-user="yilingma17@gmail.com"

source activate py39

python3.9 Ridge_train_piCTRL_lon_batch$lon_batch.py

ENDFILE

chmod +x job_ridge_train_piCTRL
sbatch job_ridge_train_piCTRL

@ lon_batch = $lon_batch + 1

end
endif
