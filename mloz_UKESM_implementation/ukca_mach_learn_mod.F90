! *****************************COPYRIGHT*******************************
! (C) Crown copyright Met Office. All rights reserved.
! For further details please refer to the file COPYRIGHT.txt
! which you should have received as part of this distribution.
! *****************************COPYRIGHT*******************************
!
! Description:
!  Calculate an ozone field from a set of coefficients provided by input file
!
!  Part of the UKCA model, a community model supported by the
!  Met Office and NCAS, with components provided initially
!  by The University of Cambridge, University of Leeds and
!  The Met. Office.  See www.ukca.ac.uk
!
! Method:
!
! Code Owner: Please refer to the UM file CodeOwners.txt
! This file belongs in section: UKCA
!
!
!  Code Description:
!   Language:  FORTRAN 90
!   This code is written to UMDP3 v6 programming standards.
!
! ----------------------------------------------------------------------
!
MODULE ukca_mach_learn_mod

USE um_types,               ONLY: integer32, real32  ! For NetCDF arguments
USE netcdf
USE um_parcore,             ONLY: mype, nproc
USE field_types,            ONLY: fld_type_p
USE UM_Parvars,             ONLY: gc_all_proc_group
USE UM_ParParams,           ONLY: halo_type_no_halo
USE mpl,                    ONLY: mpl_character, mpl_integer, mpl_real
USE ereport_mod,            ONLY: ereport
USE errormessagelength_mod, ONLY: errormessagelength
USE umPrintMgr,             ONLY: umPrint, umMessage, PrStatus_Diag, PrintStatus
USE setup_namelist,         ONLY: setup_nml_type
USE parkind1,               ONLY: jpim, jprb         ! DrHook
USE yomhook,                ONLY: lhook, dr_hook     ! DrHook

IMPLICIT NONE

PRIVATE

PUBLIC :: ukca_mach_learn

! Local variables
!
CHARACTER(LEN=*), PARAMETER, PRIVATE :: ModuleName='UKCA_MACH_LEARN_MOD'

! number of ML coefficients
INTEGER, SAVE :: n_coeffs=0

! size of ML arrays in Z (76, i.e. 50km)
INTEGER, SAVE :: n_z_pnts=76

! arrays containing ML coeffs and ozone scalings
REAL, ALLOCATABLE, SAVE :: ml_coeffs      (:,:,:,:)
REAL, ALLOCATABLE, SAVE :: ml_ozone_mean  (:,:,:)
REAL, ALLOCATABLE, SAVE :: ml_ozone_scale (:,:,:)
REAL, ALLOCATABLE, SAVE :: ml_temp_mean   (:,:,:)
REAL, ALLOCATABLE, SAVE :: ml_temp_scale  (:,:,:)


INTEGER (KIND=integer32), SAVE, PUBLIC :: nfid     ! ID of NetCDF file
INTEGER                        :: ecode            ! return code for ereport
INTEGER, PARAMETER             :: max_dims = 4  ! Max nr of dimens per variable

! Expected max No. of values in any attribute, just in case some attributes
! may be arrays (could be changed to higher value if needed)
INTEGER, PARAMETER             :: max_att_len = 10

! Expected max. length of a variable name in a NetCDF file
INTEGER, PARAMETER             :: varname_len = 80

! Lenght for the strings indicating variable type (e.g. 'CHAR', 'REAL', etc.)
INTEGER, PARAMETER             :: vartype_len = 4

INTERFACE ml_get_data
MODULE PROCEDURE            &
  ml_get_data_real3D,       &
  ml_get_data_real4D
END INTERFACE

INTERFACE ml_get_var_att
MODULE PROCEDURE            &
  ml_get_var_att_char,      &
  ml_get_var_att_int,       &
  ml_get_var_att_real
END INTERFACE

INTERFACE ml_var_check_dims
MODULE PROCEDURE                  &
  ml_var_check_dims_3d,           &
  ml_var_check_dims_4d
END INTERFACE

CONTAINS

SUBROUTINE ukca_mach_learn(row_length, rows, model_levels, temperature,         &
                           ml_ozone_out, ml_temp_out, ml_adjust_temp_out)
USE model_time_mod, ONLY: i_day_number
USE ukca_constants, ONLY: c_o3
USE missing_data_mod, ONLY: rmdi
IMPLICIT NONE

INTEGER, INTENT(IN)  :: row_length, rows, model_levels
REAL,    INTENT(IN)  :: temperature(row_length, rows, model_levels)
REAL,    INTENT(INOUT) :: ml_ozone_out(row_length, rows, model_levels)
REAL,    INTENT(OUT) :: ml_temp_out(row_length, rows, model_levels)
REAL,    INTENT(OUT) :: ml_adjust_temp_out(row_length, rows, model_levels)

! equivalent internal arrays up to size n_z_pnts (76 currently)
REAL, ALLOCATABLE, SAVE :: ml_ozone(:,:,:)
REAL, ALLOCATABLE, SAVE :: ml_temp(:,:,:)
REAL, ALLOCATABLE, SAVE :: ml_adjust_temp(:,:,:)

REAL, ALLOCATABLE, SAVE :: daily_mean_temp(:,:,:)
INTEGER, SAVE :: current_day_number=-1
INTEGER, SAVE :: n_times=0
LOGICAL, SAVE :: l_first=.TRUE.

INTEGER :: i,j

REAL, ALLOCATABLE :: tvec(:)
REAL, ALLOCATABLE :: mlmat(:,:)
REAL, ALLOCATABLE :: ml_ozone_pred(:)

! minimum ozone value allowed to trap negatives etc.
REAL, PARAMETER :: ozone_min_val=1.0e-15

INTEGER(KIND=jpim), PARAMETER :: zhook_in  = 0
INTEGER(KIND=jpim), PARAMETER :: zhook_out = 1
REAL(KIND=jprb)               :: zhook_handle

CHARACTER(LEN=*), PARAMETER :: RoutineName='UKCA_MACH_LEARN'

IF (lhook) CALL dr_hook(ModuleName//':'//RoutineName,zhook_in,zhook_handle)

! set these to zero - this will remain above n_z_pnts (76 currently)
ml_temp_out        = 0.0
ml_adjust_temp_out = 0.0
! set ozone out to missing data as this should be overwritten
! in the routine below or above 50km in ukca_main
ml_ozone_out       = rmdi

! On the first timestep set the value of current_day_number
! as cannot do ML ozone on first timestep as don't have daily
! mean temperature to work with.
! Initialise daily_mean_temp.
! Get ML coeffs and scalings. Only need to do on first timestep as these
! do not change during the run.
IF (l_first) THEN
   current_day_number     = i_day_number

   ! allocate and initialise ml_ozone
   IF (.NOT. ALLOCATED(ml_ozone))                                               &
        ALLOCATE(ml_ozone(1:row_length, 1:rows, 1:n_z_pnts))
   ml_ozone(:,:,:) = 0.0

   ! allocate and initialise ml_temp diags to zero
   IF (.NOT. ALLOCATED(ml_temp))                                                &
        ALLOCATE(ml_temp(1:row_length, 1:rows, 1:n_z_pnts))
   IF (.NOT. ALLOCATED(ml_adjust_temp))                                         &
        ALLOCATE(ml_adjust_temp(1:row_length, 1:rows, 1:n_z_pnts))
   ml_temp(:,:,:) = 0.0
   ml_adjust_temp(:,:,:) = 0.0

   IF (.NOT. ALLOCATED(daily_mean_temp))                                        &
        ALLOCATE(daily_mean_temp(1:row_length, 1:rows, 1:n_z_pnts))
   daily_mean_temp(:,:,:) = 0.0

   ! currently number of coeffs is equal to the number of levels (n_z_pnts). 
   ! May need to be namelist input eventually
   n_coeffs = n_z_pnts 
   IF (.NOT. ALLOCATED(ml_coeffs))                                              &
        ALLOCATE(ml_coeffs(1:row_length, 1:rows, 1:n_z_pnts, 1:n_coeffs))
   ml_coeffs(:,:,:,:) = 0.0
   CALL get_ml_coeffs(row_length, rows)

   ! ozone scalings to convert to correct units
   IF (.NOT. ALLOCATED(ml_ozone_mean))                                          &
        ALLOCATE(ml_ozone_mean(1:row_length, 1:rows, 1:n_z_pnts))
   ml_ozone_mean(:,:,:) = 0.0
   IF (.NOT. ALLOCATED(ml_ozone_scale))                                         &
        ALLOCATE(ml_ozone_scale(1:row_length, 1:rows, 1:n_z_pnts))
   ml_ozone_scale(:,:,:) = 0.0
   !CALL get_ml_scaling(row_length, rows)

   ! temp scalings to convert to correct units
   IF (.NOT. ALLOCATED(ml_temp_mean))                                           &
        ALLOCATE(ml_temp_mean(1:row_length, 1:rows, 1:n_z_pnts))
   ml_temp_mean(:,:,:) = 0.0
   IF (.NOT. ALLOCATED(ml_temp_scale))                                          &
        ALLOCATE(ml_temp_scale(1:row_length, 1:rows, 1:n_z_pnts))
   ! set to 1.0 here to account for division later on
   ml_temp_scale(:,:,:) = 1.0

   ! get scalings for ozone and temperature
   CALL get_ml_scaling(row_length, rows)

   l_first                = .FALSE.
END IF

! DO ML CALCULATION IF THE DAY NUMBER CHANGES
IF (current_day_number /= i_day_number) THEN
   ! calculate daily mean temperature
   daily_mean_temp(:,:,:) = daily_mean_temp(:,:,:) / n_times

   ! save temperature for output
   ml_temp = daily_mean_temp

   ! apply scaling to temperature as well
   ! this may result in negative or small values, but this is 
   ! fine given how it is processed into ozone below
   ! ml_temp_scale=1.0 and ml_temp_mean=0.0 above level 76
   daily_mean_temp(:,:,:) = (daily_mean_temp(:,:,:) - ml_temp_mean(:,:,:))      &
                            / ml_temp_scale(:,:,:)

   ! save adjusted temperature for output
   ml_adjust_temp = daily_mean_temp

   ! calculate a new value of ML_OZONE here. If not calculated it 
   ! will not change
   IF (.NOT. ALLOCATED(tvec)) ALLOCATE(tvec(1:n_z_pnts))
   IF (.NOT. ALLOCATED(mlmat)) ALLOCATE(mlmat(1:n_z_pnts,1:n_coeffs))
   IF (.NOT. ALLOCATED(ml_ozone_pred)) ALLOCATE(ml_ozone_pred(1:n_z_pnts))

   DO j=1,rows
      DO i=1,row_length
         ! Calculate the predicted ozone
         tvec(:)    = daily_mean_temp(i,j,:)
         mlmat(:,:) = ml_coeffs(i,j,:,:)
         ml_ozone_pred(:) = MATMUL(mlmat(:,:),tvec(:))
         ! rescale using provided coeffs
         ml_ozone(i,j,:) = ( ( ml_ozone_pred(:)*ml_ozone_scale(i,j,:) )         &
                           + ml_ozone_mean(i,j,:) )
      END DO
   END DO

   ! reset to minimum value
   WHERE(ml_ozone < ozone_min_val) ml_ozone = ozone_min_val

   DEALLOCATE(tvec)
   DEALLOCATE(mlmat)
   DEALLOCATE(ml_ozone_pred)

   ! at end of ML calculation update value of current_day_number
   ! to enable calculation of daily mean
   current_day_number = i_day_number
   ! reset value of n_times & daily_mean_temp
   n_times = 0
   daily_mean_temp(:,:,:) = 0.0
END IF

! if we are in the current day then start calculating daily mean temperature
! daily_mean_temp array up to height n_z_points
IF (current_day_number == i_day_number) THEN
   daily_mean_temp(:,:,:) = daily_mean_temp(:,:,:) + temperature(:,:,1:n_z_pnts)
   n_times = n_times + 1
END IF

! set output fields
ml_temp_out(:,:,1:n_z_pnts)        = ml_temp
ml_adjust_temp_out(:,:,1:n_z_pnts) = ml_adjust_temp
ml_ozone_out(:,:,1:n_z_pnts)       = ml_ozone


IF (lhook) CALL dr_hook(ModuleName//':'//RoutineName,zhook_out,zhook_handle)
RETURN

END SUBROUTINE ukca_mach_learn

SUBROUTINE get_ml_coeffs(row_length, rows)
USE ukca_option_mod, ONLY: ukca_ml_coeffs
IMPLICIT NONE

INTEGER, INTENT(IN)  :: row_length, rows

CHARACTER(LEN=varname_len), PARAMETER :: var_name='coefs'
LOGICAL :: l_norec

INTEGER :: fileid

! input data only goes up 76 levels (50km), currently n_coefs=n_z_pnts used
REAL :: ml_input_coeffs (1:row_length,1:rows,1:n_z_pnts,1:n_z_pnts)

INTEGER(KIND=jpim), PARAMETER :: zhook_in  = 0
INTEGER(KIND=jpim), PARAMETER :: zhook_out = 1
REAL(KIND=jprb)               :: zhook_handle

CHARACTER(LEN=*), PARAMETER :: RoutineName='UKCA_ML_COEFFS'

IF (lhook) CALL dr_hook(ModuleName//':'//RoutineName,zhook_in,zhook_handle)

CALL ml_fopen(TRIM(ADJUSTL(ukca_ml_coeffs)),fileid)

CALL ml_get_data (row_length, rows, fileid, TRIM(ADJUSTL(var_name)),            &
                  ml_input_coeffs, l_norec)

! only 76 levels provided in the file
ml_coeffs(:,:,:,:) = ml_input_coeffs(:,:,:,:)

CALL ml_fclose(fileid)

IF (lhook) CALL dr_hook(ModuleName//':'//RoutineName,zhook_out,zhook_handle)
END SUBROUTINE get_ml_coeffs

!----------

SUBROUTINE get_ml_scaling(row_length, rows)
USE ukca_option_mod, ONLY: ukca_ml_ozone_scaling, ukca_ml_temp_scaling
IMPLICIT NONE

INTEGER, INTENT(IN)  :: row_length, rows

CHARACTER(LEN=varname_len), PARAMETER :: t_var_mean ='x_mean'
CHARACTER(LEN=varname_len), PARAMETER :: t_var_scale='x_scale'
CHARACTER(LEN=varname_len), PARAMETER :: o3_var_mean ='y_mean'
CHARACTER(LEN=varname_len), PARAMETER :: o3_var_scale='y_scale'
LOGICAL :: l_norec

INTEGER :: fileid

INTEGER(KIND=jpim), PARAMETER :: zhook_in  = 0
INTEGER(KIND=jpim), PARAMETER :: zhook_out = 1
REAL(KIND=jprb)               :: zhook_handle

! input data only goes up 76 levels (50km)
REAL :: ml_mean  (1:row_length,1:rows,1:n_z_pnts)
REAL :: ml_scale (1:row_length,1:rows,1:n_z_pnts)


CHARACTER(LEN=*), PARAMETER :: RoutineName='GET_ML_SCALING'

IF (lhook) CALL dr_hook(ModuleName//':'//RoutineName,zhook_in,zhook_handle)

ml_mean  = 0.0
ml_scale = 0.0

! y_out = y_pred*y_scale+y_mean

! GET OZONE DATA
CALL ml_fopen(TRIM(ADJUSTL(ukca_ml_ozone_scaling)),fileid)

! mean of ozone
CALL ml_get_data (row_length, rows, fileid, TRIM(ADJUSTL(o3_var_mean)),         &
                  ml_mean, l_norec)

! only 76 levels provided in the file
ml_ozone_mean(:,:,:) = ml_mean(:,:,:)

! ozone scaling
CALL ml_get_data (row_length, rows, fileid, TRIM(ADJUSTL(o3_var_scale)),        &
                  ml_scale, l_norec)

! only 76 levels provided in the file
ml_ozone_scale(:,:,:) = ml_scale(:,:,:)

CALL ml_fclose(fileid)
!-----

! GET TEMPERATURE DATA
CALL ml_fopen(TRIM(ADJUSTL(ukca_ml_temp_scaling)),fileid)

! mean of temp
CALL ml_get_data (row_length, rows, fileid, TRIM(ADJUSTL(t_var_mean)),         &
                  ml_mean, l_norec)

! only 76 levels provided in the file
ml_temp_mean(:,:,:) = ml_mean(:,:,:)

! temp scaling
CALL ml_get_data (row_length, rows, fileid, TRIM(ADJUSTL(t_var_scale)),        &
                  ml_scale, l_norec)

! only 76 levels provided in the file
ml_temp_scale(:,:,:) = ml_scale(:,:,:)

CALL ml_fclose(fileid)
!-----

IF (lhook) CALL dr_hook(ModuleName//':'//RoutineName,zhook_out,zhook_handle)
END SUBROUTINE get_ml_scaling


!-------
! NetCDF routines for reading-in Machine Learning coefficients and ozone scalings



! ------------------------------------------------------------------------
! Interface blocks used for overloading. 
!
! A generic call to ML_GET_DATA will select the appropriate subroutine 
! automatically, depending on the type of the INTENT(INOUT) argument
! (variable values).
!
! A generic call to ML_GET_VAR_ATT will select the appropriate subroutine
! automatically, depending on the type of the INTENT(OUT) argument 
! (attribute value).
! ------------------------------------------------------------------------

! ------------------------------------------------------------------------
! Description:
!   Check for existence of NetCDF ML file and open it.
!
! Method:
!   Call NF90_OPEN to open a NetCDF file and return nstatus.
!   Call ND_ERROR, which will stop the model if nstatus reports errors.
! ------------------------------------------------------------------------
SUBROUTINE ml_fopen (filename, fileid)

IMPLICIT NONE

! Subroutine arguments
CHARACTER (LEN=*), INTENT(IN)  :: filename

INTEGER,           INTENT(OUT) :: fileid

! Local variables
INTEGER (KIND=integer32)       :: nstatus       ! netCDF return code

INTEGER (KIND=jpim), PARAMETER :: zhook_in  = 0
INTEGER (KIND=jpim), PARAMETER :: zhook_out = 1
REAL    (KIND=jprb)            :: zhook_handle

CHARACTER(LEN=*), PARAMETER :: RoutineName='ML_FOPEN'

! End of header

IF (lhook) CALL dr_hook(ModuleName//':'//RoutineName, zhook_in, zhook_handle)

IF ( mype == 0 ) THEN
  nstatus = nf90_open (filename, nf90_nowrite, nfid)
  CALL nd_error (nstatus, 'ML_FOPEN', TRIM(filename))
  fileid = nfid
ELSE
  ! Set file-id to dummy on other PEs, as they
  ! would not access the files anyway
  fileid = 99  
END IF

IF (lhook) CALL dr_hook(ModuleName//':'//RoutineName, zhook_out, zhook_handle)

RETURN
END SUBROUTINE ml_fopen

! --------------------------------------------------------------------------
! Description:
!   Obtain the following information on a NetCDF file whose id has been
!   provided: number of variables, number of dimensions, number of (global)
!   attributes
!
! Method:
!    Call NF90_INQUIRE and return only the information required. The data
!    is read on PEO and broadcast to all PEs. 
! --------------------------------------------------------------------------
SUBROUTINE ml_get_file_info (fileid,  num_vars, num_dims, num_att)

IMPLICIT NONE

! Subroutine arguments

INTEGER,           INTENT(IN)    :: fileid          ! File id

INTEGER, OPTIONAL, INTENT(OUT)   :: num_vars        ! Nr of variables
INTEGER, OPTIONAL, INTENT(OUT)   :: num_dims        ! Nr of dimensions
INTEGER, OPTIONAL, INTENT(OUT)   :: num_att         ! Nr of Attributes

! Local variables
INTEGER (KIND=integer32) :: nvars32, ndims32, natts32, undimid32
INTEGER (KIND=integer32) :: nstatus ! netCDF return code

! For Parallel I/O
INTEGER :: my_comm
INTEGER :: mpl_nml_type
INTEGER :: icode

INTEGER, PARAMETER :: no_of_types = 1 ! integer
INTEGER, PARAMETER :: n_int = 3       ! num_vars & num_dims & num_att

! Derived type used to hold information about a NetCDF file
! (read on PEO and broadcast to all PEs).
TYPE nc_fileinfo
  SEQUENCE
  INTEGER :: num_vars
  INTEGER :: num_dims
  INTEGER :: num_att
END TYPE nc_fileinfo

TYPE (nc_fileinfo) :: file_info

INTEGER (KIND=jpim), PARAMETER :: zhook_in  = 0
INTEGER (KIND=jpim), PARAMETER :: zhook_out = 1
REAL    (KIND=jprb)            :: zhook_handle

CHARACTER(LEN=*), PARAMETER :: RoutineName='ML_GET_FILE_INFO'

! End of header

IF (lhook) CALL dr_hook(ModuleName//':'//RoutineName, zhook_in, zhook_handle)

! Obtain the communication group for this run
CALL gc_get_communicator (my_comm, icode)

! Obtain an object representing the multi-datatype structure which is
! going to be broadcast. This is necessary as the broadcast operation
! needs to know exactly how many bytes of data need to be transferred.
! SETUP_NML_TYPE uses the number of variables and their size to calculate
! the resulting number of bytes as an MPL datatype.
CALL setup_nml_type (no_of_types, mpl_nml_type, n_int_in = n_int)

! PE0 reads data from the file and adds it to the derived type file_info
IF ( mype == 0 ) THEN
  nfid = fileid

  ! Get all file information anyway
  nstatus = nf90_inquire ( nfid, ndims32, nvars32, natts32, undimid32 ) 
  CALL nd_error (nstatus, 'ML_GET_FILE_INFO','INQUIRE')

  file_info % num_vars = nvars32
  file_info % num_dims = ndims32
  file_info % num_att  = natts32
END IF

! Data structure broadcast to all processors
icode = 0
CALL mpl_bcast (file_info, 1, mpl_nml_type, 0, my_comm, icode)
IF ( icode /= 0 ) THEN
  ecode = fileid
  CALL ereport ('ML_GET_FILE_INFO', ecode, 'NML Broadcast Error')
END IF

! Local variables have been filled on PE0. Now copy
! data structure (from the derived type file_info)
! to the local variables for the rest of PEs.
IF ( mype /= 0 ) THEN
  nvars32   = file_info % num_vars
  ndims32   = file_info % num_dims
  natts32   = file_info % num_att
END IF

! Check what information is required and transfer data from
! the local variables to the output arguments. Operation
! done on all PEs.
IF ( PRESENT(num_vars) ) num_vars = nvars32
IF ( PRESENT(num_dims) ) num_dims = ndims32
IF ( PRESENT(num_att ) ) num_att  = natts32

! Reset the multi-datatype structure allocated for broadcasting
CALL mpl_type_free (mpl_nml_type, icode)

IF (lhook) CALL dr_hook(ModuleName//':'//RoutineName, zhook_out, zhook_handle)

RETURN
END SUBROUTINE ml_get_file_info

! --------------------------------------------------------------------------
! Description:
!   Obtain the following information on a variable (in a NetCDF file)
!   whose name or id has been provided: variable id or name, number of
!   dimensions and corresponding sizes, and variable type.
!
! Method:
! 1) Check whether name or ID of variable in NetCDF file is supplied
!    (if name is supplied then get the id). Store id as varid_local and
!    pass it to NF90_INQUIRE_VARIABLE to get info on the variable.
! 2) Loop through all dimensions in the variable and call
!    NF90_INQUIRE_DIMENSION for each one to get their size.
! --------------------------------------------------------------------------
SUBROUTINE ml_get_var_info (ncid,    varid, varnam,                   &
                            ndims, vdimsize, var_type)

IMPLICIT NONE

! Subroutine arguments

INTEGER,           INTENT(IN)    :: ncid                ! File id

INTEGER,           INTENT(INOUT) :: varid               ! Variable id

CHARACTER (LEN=*), INTENT(INOUT) :: varnam              ! Variable name

INTEGER, OPTIONAL, INTENT(OUT)   :: ndims               ! Nr of dims
INTEGER, OPTIONAL, INTENT(OUT)   :: vdimsize(max_dims)  ! Size of dims

CHARACTER(LEN=vartype_len), OPTIONAL, INTENT(OUT) :: var_type ! Variable type

! Local variables
INTEGER (KIND=integer32)    :: nstatus        ! netCDF return code
INTEGER (KIND=integer32)    :: varid_local, itype, idumm, n, undid
INTEGER (KIND=integer32)    :: ndim_local
INTEGER (KIND=integer32)    :: dimid       (max_dims)
INTEGER (KIND=integer32)    :: dsize_local (max_dims)

CHARACTER (LEN=vartype_len) :: var_type_local
CHARACTER (LEN=varname_len) :: varnam_local

! For Parallel I/O
INTEGER :: my_comm
INTEGER :: mpl_nml_type
INTEGER :: icode

INTEGER, PARAMETER :: no_of_types = 3           ! int + logical + char
INTEGER, PARAMETER :: n_int   = 2 + max_dims    ! varid & ndims & max_dims
INTEGER, PARAMETER :: n_log = 1                 ! l_exist

! Nr. of characters for varname and var_type (see below)
INTEGER, PARAMETER :: n_chars = varname_len + vartype_len

! Derived type used to hold information about a variable in a NetCDF file
! (read on PEO and broadcast to all PEs).
TYPE nc_fileinfo
  SEQUENCE
  INTEGER                    :: varid
  INTEGER                    :: ndims
  INTEGER                    :: vdimsize(max_dims)
  CHARACTER(LEN=varname_len) :: varname
  CHARACTER(LEN=vartype_len) :: var_type
END TYPE nc_fileinfo

TYPE (nc_fileinfo) :: file_info

INTEGER (KIND=jpim), PARAMETER :: zhook_in  = 0
INTEGER (KIND=jpim), PARAMETER :: zhook_out = 1
REAL    (KIND=jprb)            :: zhook_handle

CHARACTER(LEN=*), PARAMETER :: RoutineName='ML_GET_VAR_INFO'

! End of header

IF (lhook) CALL dr_hook(ModuleName//':'//RoutineName, zhook_in, zhook_handle)

! Obtain the communication group for this run
CALL gc_get_communicator(my_comm, icode)

! Obtain an object representing the multi-datatype structure which is
! going to be broadcast (see further details above)
CALL setup_nml_type(no_of_types, mpl_nml_type, n_int_in = n_int,    &
                    n_log_in = n_log, n_chars_in = n_chars )

! Check whether Name or ID of variable in NetCDF file is supplied.
! PE0 reads the relevant data and adds it to the derived type file_info.
IF ( mype == 0 ) THEN
  nfid = ncid

  ! The following lines include NetCDF calls to populate local variables
  ! (e.g. varid_local, ndim_local, dsize_local(:), varnam_local,
  ! var_type_local) which will be transferred to the broadcast type.
  ! There is no need to initialise most of those local variables here,
  ! because if they were not correctly populated the NetCDF calls would
  ! report errors anyway.

  ! If varnam provided then get varid
  IF ( TRIM(varnam) /= "" ) THEN
    ! Note: 'varnam' is intent(in) in nf90_inq_varid.
    ! See note below about 'varnam_local'.
    nstatus = nf90_inq_varid (nfid, TRIM(varnam), varid_local)
    CALL nd_error ( nstatus, 'ML_GET_VAR_INFO', 'INQ_VARID ' // varnam )
    varid = varid_local

  ! If varnam not provided then make sure that varid has
  ! been provided; if that is not the case then report error.
  ELSE
    IF ( varid <= 0 ) THEN
      ecode = 1
      CALL ereport ('ML_GET_VAR_INFO', ecode,           &
        'Either variable name or ID should be provided')
    END IF
    varid_local = varid

  END IF            ! check if name/id provided

  ! Get type information, dimension ids, etc.
  ! 
  ! Here we get the name of the variable, now 'varnam_local',
  ! which unlike 'varnam' above needs to be intent(out) in
  ! nf90_inquire_variable. That is the reason why two
  ! 'varnams' are used. It is necessary to get 'varnam_local'
  ! because this routine can be called with 'varnam' empty.
  !
  ! Note also that if the user did pass in 'varnam' it will be
  ! overwritten by 'varnam_local', but that is OK because 
  ! the code above has ensured that the variable name 
  ! corresponds to the id being inquired about.
  !
  nstatus = nf90_inquire_variable (nfid,  varid_local, varnam_local, itype, &
                                   ndim_local, dimid, idumm)

  CALL nd_error (nstatus, 'ML_GET_VAR_INFO',                                & 
                'INQ_VARIABLE ' // TRIM(varnam_local) )
  varnam = varnam_local

  ! Get size of each dimension
  IF ( PRESENT(vdimsize) ) THEN
    DO n = 1, ndim_local
      nstatus = nf90_inquire_dimension (nfid, dimid(n), LEN = dsize_local(n))
      CALL nd_error (nstatus, 'ML_GET_VAR_INFO', 'INQ_DIMSIZE ' // TRIM(varnam))
    END DO
    vdimsize (1:ndim_local) = dsize_local (1:ndim_local)
  END IF

  ! Convert type into string (only the following common types for now)
  IF ( PRESENT(var_type) ) THEN
    SELECT CASE(itype)
    CASE (nf90_char)
      var_type_local = 'CHAR'
    CASE (nf90_int)
      var_type_local = 'INT'
    CASE (nf90_float)
      var_type_local = 'REAL'
    CASE (nf90_double)
      var_type_local = 'DBLE'
    CASE DEFAULT
      WRITE (umMessage,'(A,1X,I2)') 'ML_GET_VAR_INFO: ' //               &
                       'Unknown variable type defined ', itype
      CALL umPrint (umMessage, src='ml_get_var_info')
      WRITE (umMessage,'(A)')  'Should be one of: CHAR / INT / REAL / DBLE'
      CALL umPrint (umMessage, src='ml_get_var_info')
      nstatus = nf90_ebadname
      CALL nd_error (nstatus, 'ML_GET_VAR_INFO', 'VAR_TYPE ' // TRIM(varnam))
    END SELECT
  END IF 

  ! Fill in derived type variable to later do broadcast
  file_info % varid       = varid_local
  file_info % ndims       = ndim_local
  file_info % vdimsize(:) = dsize_local(:)
  file_info % varname     = varnam_local
  file_info % var_type    = var_type_local

END IF

! Data structure broadcast to all processors
icode = 0
CALL mpl_bcast (file_info, 1, mpl_nml_type, 0, my_comm, icode)
IF ( icode /= 0 ) THEN
  ecode = ncid
  CALL ereport ('ML_GET_VAR_INFO', ecode, 'NML Broadcast Error')
END IF

! Local variables have been filled on PE0. Now transfer
! data from the broadcast type into the local variables
! for the rest of PEs.
IF ( mype /= 0 ) THEN
  varid_local    = file_info % varid
  varnam_local   = file_info % varname
  ndim_local     = file_info % ndims
  dsize_local(:) = file_info % vdimsize(:)
  var_type_local = file_info % var_type
END IF

! Copy the local variables back onto the output variables
varid  = varid_local
varnam = varnam_local
IF ( PRESENT(ndims) )    ndims       = ndim_local
IF ( PRESENT(vdimsize) ) vdimsize(:) = dsize_local(:)
IF ( PRESENT(var_type) ) var_type    = var_type_local

! Reset the multi-datatype structure allocated for broadcasting
CALL mpl_type_free (mpl_nml_type, icode)

IF (lhook) CALL dr_hook(ModuleName//':'//RoutineName, zhook_out, zhook_handle)

RETURN
END SUBROUTINE ml_get_var_info


! --------------------------------------------------------------------------
! Description:
!   Check that dimensions of a variable in a NetCDF file match global sizes,
!   in the case of a full-atmosphere three-dimensional distribution.
!
! Method:
! 1) Call ML_GET_VAR_INFO to get the dimensions and check they match.
! 2) If wrong dimensions then stop with call to EREPORT
! --------------------------------------------------------------------------
SUBROUTINE ml_var_check_dims_3d (global_row_length, global_rows, model_levels, &
                                 fileid, varname, dimsize)

IMPLICIT NONE

! Subroutine arguments
INTEGER,           INTENT(IN)    :: global_row_length
INTEGER,           INTENT(IN)    :: global_rows
INTEGER,           INTENT(IN)    :: model_levels
INTEGER,           INTENT(IN)    :: fileid

CHARACTER (LEN=*), INTENT(INOUT) :: varname

INTEGER,           INTENT(OUT)   :: dimsize(max_dims)

! Local variables
INTEGER                        :: vid    ! variable id

INTEGER (KIND=jpim), PARAMETER :: zhook_in  = 0
INTEGER (KIND=jpim), PARAMETER :: zhook_out = 1
REAL    (KIND=jprb)            :: zhook_handle

CHARACTER(LEN=*), PARAMETER :: RoutineName='ML_VAR_CHECK_DIMS_3D'

! End of header

IF (lhook) CALL dr_hook(ModuleName//':'//RoutineName, zhook_in, zhook_handle)

! Emission files should be defined as (x,y,lev,t)
!
! Get information on variable. One of the two INOUT arguments
! of ML_GET_VAR_INFO is supplied (varname); assign an arbitrary
! negative value to the other one since it is not known yet (vid).
vid = -1
CALL ml_get_var_info (fileid, vid, varname, vdimsize=dimsize)

! Check that dimensions (x, y, lev) match those of the model
IF ( dimsize(1) /= global_row_length .OR. dimsize(2) /= global_rows .OR.  &
    (dimsize(3) > 1 .AND. dimsize(3) /= model_levels)               .OR.  &
     dimsize(3) < 1 ) THEN  

  WRITE (umMessage,'(A,A)') 'ERROR: Variable dimensions do not match for ',&
                   TRIM(varname)
  CALL umPrint(umMessage,src='ml_var_check_dims')
  WRITE (umMessage,'(A,3(1x,I5))') 'Expected:', global_row_length,  &
                    global_rows, model_levels
  CALL umPrint(umMessage,src='ml_var_check_dims')
  WRITE (umMessage,'(A,3(1x,I5))') 'Found:', dimsize(1:3)
  CALL umPrint(umMessage,src='ml_var_check_dims')
  ecode = fileid
  CALL ereport ('ML_VAR_CHECK_DIMS_3D', ecode,                            &
      'Dimension mismatch for: ' // TRIM(varname))
END IF

IF (lhook) CALL dr_hook(ModuleName//':'//RoutineName, zhook_out, zhook_handle)

RETURN
END SUBROUTINE ml_var_check_dims_3d

! --------------------------------------------------------------------------
! Description:
!   Check that dimensions of a variable in a NetCDF file match global sizes,
!   in the case of a full-atmosphere 3D distribution.
!
! Method:
! 1) Call ML_GET_VAR_INFO to get the dimensions and check they match.
! 2) If wrong dimensions then stop with call to EREPORT
! --------------------------------------------------------------------------
SUBROUTINE ml_var_check_dims_4d (global_row_length, global_rows, model_levels, &
                                 dimension_4, fileid, varname, dimsize)

IMPLICIT NONE

! Subroutine arguments
INTEGER,           INTENT(IN)    :: global_row_length
INTEGER,           INTENT(IN)    :: global_rows
INTEGER,           INTENT(IN)    :: model_levels
INTEGER,           INTENT(IN)    :: dimension_4
INTEGER,           INTENT(IN)    :: fileid

CHARACTER (LEN=*), INTENT(INOUT) :: varname

INTEGER,           INTENT(OUT)   :: dimsize(max_dims)

! Local variables
INTEGER                        :: vid    ! variable id

INTEGER (KIND=jpim), PARAMETER :: zhook_in  = 0
INTEGER (KIND=jpim), PARAMETER :: zhook_out = 1
REAL    (KIND=jprb)            :: zhook_handle

CHARACTER(LEN=*), PARAMETER :: RoutineName='ML_VAR_CHECK_DIMS_4D'

! End of header

IF (lhook) CALL dr_hook(ModuleName//':'//RoutineName, zhook_in, zhook_handle)

! 4D emission files should be defined as (x,y,lev,dimension_4,t)
!
! Get information on variable. One of the two INOUT arguments
! of ML_GET_VAR_INFO is supplied (varname); assign an arbitrary
! negative value to the other one since it is not known yet (vid).
vid = -1
CALL ml_get_var_info (fileid, vid, varname, vdimsize=dimsize)

! Check that dimensions (x, y, lev, dimension_4) match those of the model
IF ( dimsize(1) /= global_row_length .OR. dimsize(2) /= global_rows .OR.  &
    (dimsize(3) > 1 .AND. dimsize(3) /= model_levels)               .OR.  &
     dimsize(3) < 1                                                 .OR.  &
     dimsize(4) /= dimension_4 ) THEN

  WRITE (umMessage,'(A,A)') 'ERROR: Variable dimensions do not match for ',&
                   TRIM(varname)
  CALL umPrint(umMessage,src='ml_var_check_dims')
  WRITE (umMessage,'(A,4(1x,I0))') 'Expected:', global_row_length,  &
                    global_rows, model_levels, dimension_4
  CALL umPrint(umMessage,src='ml_var_check_dims')
  WRITE (umMessage,'(A,4(1x,I0))') 'Found:', dimsize(1:4)
  CALL umPrint(umMessage,src='ml_var_check_dims')
  ecode = fileid
  CALL ereport ('ML_VAR_CHECK_DIMS_4D', ecode,                             &
      'Dimension mismatch for: ' // TRIM(varname))
END IF

IF (lhook) CALL dr_hook(ModuleName//':'//RoutineName, zhook_out, zhook_handle)

RETURN
END SUBROUTINE ml_var_check_dims_4d

!-------------------------------------------------------------------------
! Description:
!   Read data values of three-dimensional real variable in NetCDF file.
!   Part of the interface block ML_GET_DATA used for overloading; this
!   subroutine is called when the variable is of type real 3-dim.
!
! Method:
!   Get variable ID (varid), number of dimensions in variable (ndim), 
!   and vector of dimension sizes (dim_size) by calling ML_GET_VAR_INFO.
!   Then get domain/indices to extract and call NF90_GET_VAR to get the 
!   variable values, which are read by PEO as global_var. The values
!   are broadcast to all PEs, looping over levels, and returned as rvalue.
!   Assumes (lon, lat, lev).
!-------------------------------------------------------------------------
SUBROUTINE ml_get_data_real3D (row_length, rows, fileid,   &
                               var_name, rvalue, L_norec)

USE nlsizes_namelist_mod,  ONLY:global_row_length, global_rows

IMPLICIT NONE

! Subroutine arguments
INTEGER,           INTENT (IN)    :: row_length ! model horiz dimensions
INTEGER,           INTENT (IN)    :: rows
INTEGER,           INTENT (IN)    :: fileid    ! file id

CHARACTER (LEN=*), INTENT (IN)    :: var_name  ! variable name

! variable values to read (lon, lat, lev)
REAL,              INTENT (INOUT) :: rvalue (:,:,:)

LOGICAL,           INTENT (OUT)   :: L_norec     ! T if rec not in file, EOF

! Local variables

INTEGER (KIND=integer32) :: nstatus ! netCDF return code

! Vector specifying the index in the variable from which
! the first data value will be read along each dimension.
INTEGER (KIND=integer32) :: start   (max_dims)

! Vector specifying the number of counters to read along each dimension
INTEGER (KIND=integer32) :: counter (max_dims)

INTEGER (KIND=integer32) :: varid32
INTEGER                  :: varid, ndim, dim_size(max_dims)

CHARACTER (LEN=varname_len) :: varname_local

! Variable to hold file data
REAL, ALLOCATABLE        :: global_var(:,:,:)

INTEGER:: icode, k
CHARACTER (LEN=errormessagelength) :: cmessage  ! error message

INTEGER (KIND=jpim), PARAMETER :: zhook_in  = 0
INTEGER (KIND=jpim), PARAMETER :: zhook_out = 1
REAL    (KIND=jprb)            :: zhook_handle

CHARACTER(LEN=*), PARAMETER :: RoutineName='ML_GET_DATA_REAL3D'

! End of header

IF (lhook) CALL dr_hook(ModuleName//':'//RoutineName, zhook_in, zhook_handle)

! Get information on variable (need to do this from all PEs so that
! global_var can be allocated with the right 3rd dimension)
!
! One of the two INOUT arguments of ML_GET_VAR_INFO is supplied
! (varname_local); assign an arbitrary negative value to the other 
! one since it is not known yet (varid).
varname_local = var_name    ! need to pass INOUT argument
varid         = -1
CALL ml_get_var_info (fileid,      varid,             varname_local,   &
                      ndims=ndim,  vdimsize=dim_size)

! Check that data in NetCDF file matches Global Dimensions and 
! output variable rvalue has correct space for levels
IF (dim_size (1)   /= global_row_length .OR.            &
    dim_size (2)   /= global_rows       .OR.            &
    SIZE(rvalue,3) /= dim_size(3))      THEN
    ecode = 1
    CALL ereport ('ML_GET_DATA_REAL3D', ecode,          &
       'Dimension mismatch for: ' // TRIM(var_name))
END IF

! Allocate the variable that will hold the data. It will be filled
! by PE0 and broadcast to other PEs.
ALLOCATE ( global_var (dim_size(1), dim_size(2), dim_size(3)) )

! Get values of variable from PE0 and store in global_var
IF ( mype == 0 ) THEN
  ! Default indices for the first data value and number of counters
  ! along two dimensions: Global lon and lat. Note that counter(3) will
  ! often be different from the number of model vertical levels, as
  ! it happens in the case of surface emissions.
  start   (1:3) = 1
  counter (1)   = dim_size (1)
  counter (2)   = dim_size (2)
  counter (3)   = dim_size (3)

  ! Get values of the variable as global_var, specifying what to read
  ! with the optional start and counter arguments
  nfid    = fileid
  varid32 = varid    ! 32-bit integer needed
  nstatus = nf90_get_var (nfid, varid32, global_var, start  (1:ndim),  &
                                                     counter(1:ndim))
  CALL nd_error (nstatus, 'ML_GET_DATA_REAL3D', var_name)
END IF    ! I/O PE

icode = 0
CALL gc_gsync(nproc,icode)   ! Synchronise PEs before scatter

! Distribute data to all PEs, looping over levels
DO k = 1, dim_size(3)
  ! DEPENDS ON: scatter_field
  CALL scatter_field(rvalue(:,:,k), global_var(:,:,k),       &
                   row_length, rows,                         &
                   global_row_length, global_rows,           &
                   fld_type_p, halo_type_no_halo,            &
                   0, gc_all_proc_group)
END DO

DEALLOCATE(global_var)

IF (lhook) CALL dr_hook(ModuleName//':'//RoutineName, zhook_out, zhook_handle)

RETURN
END SUBROUTINE ml_get_data_real3D

!-------------------------------------------------------------------------
! Description:
!   Read data values of four-dimensional real variable in NetCDF file.
!   Part of the interface block ML_GET_DATA used for overloading; this
!   subroutine is called when the variable is of type real 4-dim.
!
! Method:
!   Get variable ID (varid), number of dimensions in variable (ndim),
!   and vector of dimension sizes (dim_size) by calling ML_GET_VAR_INFO.
!   Then get domain/indices to extract and call NF90_GET_VAR to get the
!   variable values, which are read by PEO as global_var. The values
!   are broadcast to all PEs, looping over levels, and returned as rvalue.
!   Assumes (lon, lat, lev, dimension_4).
!-------------------------------------------------------------------------
SUBROUTINE ml_get_data_real4D (row_length, rows, fileid,   &
                               var_name, rvalue, L_norec)

USE nlsizes_namelist_mod,  ONLY:global_row_length, global_rows

IMPLICIT NONE

! Subroutine arguments
INTEGER,           INTENT (IN)    :: row_length ! model horiz dimensions
INTEGER,           INTENT (IN)    :: rows
INTEGER,           INTENT (IN)    :: fileid    ! file id

CHARACTER (LEN=*), INTENT (IN)    :: var_name  ! variable name

! variable values to read (lon, lat, lev, dimension_4)
REAL,              INTENT (INOUT) :: rvalue (:,:,:,:)

LOGICAL,           INTENT (OUT)   :: L_norec     ! T if rec not in file, EOF

! Local variables

INTEGER (KIND=integer32) :: nstatus ! netCDF return code

! Vector specifying the index in the variable from which
! the first data value will be read along each dimension.
INTEGER (KIND=integer32) :: start   (max_dims)

! Vector specifying the number of counters to read along each dimension
INTEGER (KIND=integer32) :: counter (max_dims)

INTEGER (KIND=integer32) :: varid32
INTEGER                  :: varid, ndim, dim_size(max_dims)

CHARACTER (LEN=varname_len) :: varname_local

! Variable to hold file data
REAL, ALLOCATABLE        :: global_var(:,:,:,:)

INTEGER:: icode, k, i
CHARACTER (LEN=errormessagelength) :: cmessage  ! error message

INTEGER (KIND=jpim), PARAMETER :: zhook_in  = 0
INTEGER (KIND=jpim), PARAMETER :: zhook_out = 1
REAL    (KIND=jprb)            :: zhook_handle

CHARACTER(LEN=*), PARAMETER :: RoutineName='ML_GET_DATA_REAL4D'

! End of header

IF (lhook) CALL dr_hook(ModuleName//':'//RoutineName, zhook_in, zhook_handle)

! Get information on variable (need to do this from all PEs so that
! global_var can be allocated with the right 4th dimension)
!
! One of the two INOUT arguments of ML_GET_VAR_INFO is supplied
! (varname_local); assign an arbitrary negative value to the other
! one since it is not known yet (varid).
varname_local = var_name    ! need to pass INOUT argument
varid         = -1
CALL ml_get_var_info (fileid,      varid,             varname_local,   &
                      ndims=ndim,  vdimsize=dim_size)

! Check that data in NetCDF file matches Global Dimensions and
! output variable rvalue has correct space for levels
IF (dim_size (1)   /= global_row_length .OR.            &
    dim_size (2)   /= global_rows       .OR.            &
    SIZE(rvalue,3) /= dim_size(3)       .OR.            &
    SIZE(rvalue,4) /= dim_size(4))      THEN
    ecode = 1
    CALL ereport ('ML_GET_DATA_REAL3D', ecode,          &
       'Dimension mismatch for: ' // TRIM(var_name))
END IF

! Allocate the variable that will hold the data. It will be filled
! by PE0 and broadcast to other PEs.
ALLOCATE ( global_var (dim_size(1), dim_size(2),        &
                       dim_size(3), dim_size(4)) )

! Get values of variable from PE0 and store in global_var
IF ( mype == 0 ) THEN
  ! Default indices for the first data value and number of counters
  ! along two dimensions: Global lon and lat. Note that counter(3) will
  ! often be different from the number of model vertical levels, as
  ! it happens in the case of surface emissions.
  start   (1:4) = 1
  counter (1)   = dim_size (1)
  counter (2)   = dim_size (2)
  counter (3)   = dim_size (3)
  counter (4)   = dim_size (4)

  ! Get values of the variable as global_var, specifying what to read
  ! with the optional start and counter arguments
  nfid    = fileid
  varid32 = varid    ! 32-bit integer needed
  nstatus = nf90_get_var (nfid, varid32, global_var, start  (1:ndim),  &
                                                     counter(1:ndim))
  CALL nd_error (nstatus, 'ML_GET_DATA_REAL4D', var_name)
END IF    ! I/O PE

icode = 0
CALL gc_gsync(nproc,icode)   ! Synchronise PEs before scatter

! Distribute data to all PEs, looping over levels
DO i = 1, dim_size(4)
  DO k = 1, dim_size(3)
    ! DEPENDS ON: scatter_field
    CALL scatter_field(rvalue(:,:,k,i), global_var(:,:,k,i),   &
                     row_length, rows,                         &
                     global_row_length, global_rows,           &
                     fld_type_p, halo_type_no_halo,            &
                     0, gc_all_proc_group)
  END DO ! k
END DO ! i

DEALLOCATE(global_var)

IF (lhook) CALL dr_hook(ModuleName//':'//RoutineName, zhook_out, zhook_handle)

RETURN
END SUBROUTINE ml_get_data_real4D

! --------------------------------------------------------------------------
! Description:
!   Read attributes of type character from a NetCDF file, given the
!   variable and attribute names.
!   Part of the interface block ML_GET_VAR_ATT used for overloading; this
!   subroutine is called when the required attribute is of type character.
!
! Method:
!   If a global attribute is required, variable name = 'global'.
!   If variable name is passed call NF90_INQ_VARID to get variable-id.
!   Call NF90_GET_ATT to get attribute value: cvalue.
!   Attribute values are read on PE0 and broadcast to all PEs.
! --------------------------------------------------------------------------
SUBROUTINE ml_get_var_att_char (fileid, ilen, var_name, att_name, cvalue , &
                                att_exists)

IMPLICIT NONE

!  Subroutine arguments
INTEGER,          INTENT(IN)    :: fileid    ! file id
INTEGER,          INTENT(IN)    :: ilen      ! num items in value
                                             ! (not used for chars)

CHARACTER(LEN=*), INTENT(IN)    :: var_name  ! variable name
CHARACTER(LEN=*), INTENT(IN)    :: att_name  ! attribute name

CHARACTER(LEN=*), INTENT(OUT)   :: cvalue    ! value out

! Optional argument used to check if an attribute is present. If
! attribute found then l_exist will be set to True, otherwise
! l_exist will be False but the model will not stop with error.
LOGICAL, OPTIONAL, INTENT(INOUT) :: att_exists

!  Local variables
INTEGER   (KIND=integer32) :: varid32
INTEGER   (KIND=integer32) :: nstatus          ! netCDF return code
INTEGER,  PARAMETER        :: charlength = 256
CHARACTER (LEN=charlength) :: cval             ! Dummy char to hold value
LOGICAL                    :: l_exist

! For Parallel I/O
INTEGER :: my_comm
INTEGER :: mpl_nml_type
INTEGER :: icode

INTEGER, PARAMETER :: no_of_types = 2          ! logical + char
INTEGER, PARAMETER :: n_log   = 1              ! l_exist
INTEGER, PARAMETER :: n_chars = 1 * charlength ! att_value

! Derived type used to hold attribute values from a NetCDF file
! (read on PEO and broadcast to all PEs).
TYPE nc_fileinfo
  SEQUENCE
  LOGICAL                   :: l_exist
  CHARACTER(LEN=charlength) :: att_value
END TYPE nc_fileinfo

TYPE (nc_fileinfo) :: file_info

INTEGER (KIND=jpim), PARAMETER :: zhook_in  = 0
INTEGER (KIND=jpim), PARAMETER :: zhook_out = 1
REAL    (KIND=jprb)            :: zhook_handle

CHARACTER(LEN=*), PARAMETER :: RoutineName='ML_GET_VAR_ATT_CHAR'

!  End of header
   
IF (lhook) CALL dr_hook(ModuleName//':'//RoutineName, zhook_in, zhook_handle)

! Obtain the communication group for this run
CALL gc_get_communicator(my_comm, icode)

! Obtain an object representing the multi-datatype structure which is
! going to be broadcast (see further details above)
CALL setup_nml_type(no_of_types, mpl_nml_type, n_log_in = n_log, &
                    n_chars_in = n_chars)

! PE0 reads attribute values and adds them to the derived type file_info
IF ( mype == 0 ) THEN
  nfid = fileid
  ! Get variable id
  IF ( TRIM(var_name) == 'GLOBAL' .OR. TRIM(var_name) == 'global' .OR. &
       TRIM(var_name) == 'Global' ) THEN
    varid32 = NF90_GLOBAL
  ELSE
    nstatus = nf90_inq_varid (nfid, var_name, varid32)
    CALL nd_error (nstatus, 'ML_GET_VAR_ATT_CHAR: INQ Varid', var_name)
  END IF

  ! Get value of attribute
  cval    = ' '
  nstatus = nf90_get_att (nfid, varid32, att_name, cval)

  ! If argument att_exists is present that means that the attribute is not
  ! compulsory. Therefore, if attribute found in the previous call to 
  ! nf90_get_att then set the local variable l_exist to True; otherwise 
  ! leave it as False but there is no need to flag error.
  l_exist = .FALSE.
  IF ( PRESENT(att_exists) ) THEN
    IF (nstatus == nf90_noerr) l_exist = .TRUE. ! attrib exists

  ! If argument att_exists not present then the attribute is needed. Therefore
  ! call nd_error, which will stop the model only if the previous call to
  ! nf90_get_att reported error (as given by nstatus).
  ELSE
    CALL nd_error (nstatus, 'ML_GET_VAR_ATT_CHAR: Get Attribute for ' // &
                   TRIM(var_name), att_name)    ! Check for error
  END IF

  ! Fill in output argument.
  cvalue = TRIM(cval)

  ! Copy into broadcast type
  file_info % att_value = cvalue
  file_info % l_exist   = l_exist

END IF

! Distribute data to all other PEs
icode = 0
CALL mpl_bcast (file_info, 1, mpl_nml_type, 0, my_comm, icode)
IF ( icode /= 0 ) THEN
  ecode = fileid
  CALL ereport ('ML_GET_VAR_ATT_CHAR', ecode, 'NML Broadcast Error')
END IF

! Operation on all PEs different from PE0: Transfer data from broadcast type
! into output argument cvalue and local variable l_exist.
IF ( mype /= 0 ) THEN
  cvalue  = file_info % att_value
  l_exist = file_info % l_exist
END IF

! Fill in optional argument (if present) with value from local variable.
IF ( PRESENT(att_exists) ) att_exists = l_exist

! Reset the multi-datatype structure allocated for broadcasting
CALL mpl_type_free (mpl_nml_type, icode)

IF (lhook) CALL dr_hook(ModuleName//':'//RoutineName, zhook_out, zhook_handle)

RETURN
END SUBROUTINE ml_get_var_att_char

! --------------------------------------------------------------------------
! Description:
!   Read attributes of type integer from a NetCDF file, given the
!   variable and attribute names.
!   Part of the interface block ML_GET_VAR_ATT used for overloading; this
!   subroutine is called when the required attribute is of type integer.
!   It can also convert character attributes to integers for compatibility 
!   with older files which contain integer attributes in strings, e.g. "1".
!
! Method:
!   If a global attribute is required, variable name = 'global'.
!   If variable name is passed call NF90_INQ_VARID to get variable-id.
!   Call NF90_GET_ATT to get attribute value: ivalue.
!   Attribute values are read on PE0 and broadcast to all PEs.
! --------------------------------------------------------------------------
SUBROUTINE ml_get_var_att_int (fileid, ilen, var_name, att_name, ivalue, &
                               att_exists )

IMPLICIT NONE

!  Subroutine arguments
INTEGER,          INTENT(IN)    :: fileid       ! file id
INTEGER,          INTENT(IN)    :: ilen         ! num items in value (just
                                                ! in case it is an array)

CHARACTER(LEN=*), INTENT(IN)    :: var_name     ! variable name
CHARACTER(LEN=*), INTENT(IN)    :: att_name     ! attribute name

INTEGER,          INTENT(OUT)   :: ivalue(ilen) ! value out

! Optional argument used to check if an attribute is present. If
! attribute found then l_exist will be set to True, otherwise
! l_exist will be False but the model will not stop with error.
LOGICAL, OPTIONAL, INTENT(INOUT) :: att_exists

!  Local variables
LOGICAL :: l_exist
INTEGER,  PARAMETER        :: charlength = 12
CHARACTER (LEN=charlength) :: cval             ! Dummy char to hold value

! 32-bit integers to actually pass to NC routines
INTEGER(KIND=integer32) :: varid32, ival32(ilen)
INTEGER(KIND=integer32) :: nstatus               ! netCDF return code
INTEGER(KIND=integer32) :: iattlen               ! length of attribute in file
INTEGER(KIND=integer32) :: xtype                 ! netcdf data type

! For Parallel I/O
INTEGER :: my_comm
INTEGER :: mpl_nml_type
INTEGER :: icode

INTEGER, PARAMETER :: no_of_types = 2     ! integer + logical
INTEGER, PARAMETER :: n_int = max_att_len ! max possible size
INTEGER, PARAMETER :: n_log = 1           ! l_exist

! Derived type used to hold attribute values from a NetCDF file
! (read on PEO and broadcast to all PEs).
TYPE nc_fileinfo
  SEQUENCE
  INTEGER :: att_value(max_att_len)
  LOGICAL :: l_exist
END TYPE nc_fileinfo

TYPE (nc_fileinfo) :: file_info

INTEGER (KIND=jpim), PARAMETER :: zhook_in  = 0
INTEGER (KIND=jpim), PARAMETER :: zhook_out = 1
REAL    (KIND=jprb)            :: zhook_handle

CHARACTER(LEN=*), PARAMETER :: RoutineName='ML_GET_VAR_ATT_INT'

!  End of header
   
IF (lhook) CALL dr_hook(ModuleName//':'//RoutineName, zhook_in, zhook_handle)

! Obtain the communication group for this run
CALL gc_get_communicator(my_comm, icode)

! Obtain an object representing the multi-datatype structure which is
! going to be broadcast (see further details above)
CALL setup_nml_type (no_of_types, mpl_nml_type, n_int_in = n_int, &
                     n_log_in = n_log)

! PE0 reads attribute values and adds them to the derived type file_info
IF ( mype == 0 ) THEN
  ! Get variable id
  nfid = fileid
  IF ( TRIM(var_name) == 'GLOBAL'   .OR. TRIM(var_name) == 'global' .OR. &
       TRIM(var_name) == 'Global' ) THEN
    varid32 = NF90_GLOBAL
  ELSE
    nstatus = nf90_inq_varid (nfid, var_name, varid32)
    CALL nd_error (nstatus, 'ML_GET_VAR_ATT_INT: INQ Varid', var_name)
  END IF

  ! Check data type of attribute and whether it exists
  nstatus = nf90_inquire_attribute(nfid, varid32, att_name, xtype=xtype,  &
                                   len=iattlen)

  ! Set l_exist according to whether this call returned an error (nstatus).
  ! If optional argument att_exists is present, then attribute is allowed to be
  ! missing and the value of l_exist is returned. If not, call nd_error which
  ! will stop the model since nstatus /= nf90_noerr.
  IF (nstatus == nf90_noerr) THEN
    l_exist = .TRUE.  ! attrib exists
  ELSE
    l_exist = .FALSE.
    IF (.NOT. PRESENT(att_exists) ) THEN
      CALL nd_error (nstatus, 'ML_GET_VAR_ATT_INT: Inquire Attribute for ' // &
                     TRIM(var_name), att_name)
    END IF
  END IF

  ! If attribute does exist, read it. Convert data type here if required.
  ival32(:) = 0
  IF (l_exist) THEN
    SELECT CASE(xtype)
    CASE(NF90_CHAR)
      ! Treat string as a single integer, iattlen is length of string.
      ! '(I12)' format statement must agree with charlength variable. 12 digits
      ! is enough to hold any 32-bit integer.
      cval = ' '
      nstatus = nf90_get_att (nfid, varid32, att_name, cval(1:iattlen))
      READ (cval, '(I12)') ival32(1)
      
    CASE DEFAULT
      ! All other types are numbers so netcdf will convert to int on the fly
      nstatus   = nf90_get_att (nfid, varid32, att_name, ival32(1:iattlen))

    END SELECT
    CALL nd_error (nstatus, 'ML_GET_VAR_ATT_INT: Get Attribute for ' // &
                   TRIM(var_name), att_name)
  END IF

  ! Store local variables on broadcast type
  file_info % l_exist           = l_exist
  file_info % att_value(1:ilen) = ival32(1:ilen) 

END IF

! Distribute data to all other PEs
icode = 0
CALL mpl_bcast (file_info, 1, mpl_nml_type, 0, my_comm, icode)
IF ( icode /= 0 ) THEN
  ecode = fileid
  CALL ereport ('ML_GET_ATT_INT', ecode, 'NML Broadcast Error')
END IF
  
! Operation on all PEs different from PE0: Transfer data from
! broadcast type into local variables.
IF ( mype /= 0 ) THEN
  ival32(1:ilen) = file_info % att_value(1:ilen)
  l_exist        = file_info % l_exist
END IF

! Fill in the output argument 'ivalue' and (if present) the optional
! argument 'att_exists' with the values taken from the local variables.
! Operation done on all processors.
ivalue = ival32
IF ( PRESENT(att_exists) ) att_exists = l_exist

! Reset the multi-datatype structure allocated for broadcasting
CALL mpl_type_free (mpl_nml_type, icode)

IF (lhook) CALL dr_hook(ModuleName//':'//RoutineName, zhook_out, zhook_handle)

RETURN
END SUBROUTINE ml_get_var_att_int

! --------------------------------------------------------------------------
! Description:
!   Read attributes of type real from a NetCDF file, given the
!   variable and attribute names.
!   Part of the interface block ML_GET_VAR_ATT used for overloading; this
!   subroutine is called when the required attribute is of type real.
!
! Method:
!   If a global attribute is required, variable name = 'global'.
!   If variable name is passed call NF90_INQ_VARID to get variable-id.
!   Call NF90_GET_ATT to get attribute value: rvalue.
!   Attribute values are read on PE0 and broadcast to all PEs.
! --------------------------------------------------------------------------
SUBROUTINE ml_get_var_att_real (fileid, ilen, var_name, att_name, rvalue, &
                                att_exists)

IMPLICIT NONE

!  Subroutine arguments
INTEGER,           INTENT(IN)    :: fileid       ! file id
INTEGER,           INTENT(IN)    :: ilen         ! num items in value (just in
                                                 ! case it is an array)

CHARACTER(LEN=*),  INTENT(IN)    :: var_name     ! variable name
CHARACTER(LEN=*),  INTENT(IN)    :: att_name     ! attribute name

REAL,              INTENT(OUT)   :: rvalue(ilen) ! value out

! Optional argument used to check if an attribute is present. If
! attribute found then l_exist will be set to True, otherwise
! l_exist will be False but the model will not stop with error.
LOGICAL, OPTIONAL, INTENT(INOUT) :: att_exists

!  Local variables
INTEGER(KIND=integer32) :: varid32
INTEGER(KIND=integer32) :: nstatus       ! netCDF return code
REAL   (KIND=real32)    :: rval32(ilen)
LOGICAL                 :: l_exist 

! For Parallel I/O
INTEGER :: my_comm
INTEGER :: mpl_nml_type
INTEGER :: icode

INTEGER, PARAMETER :: no_of_types = 2        ! real + logical
INTEGER, PARAMETER :: n_real = max_att_len   ! max possible size
INTEGER, PARAMETER :: n_log = 1              ! l_exist

! Derived type used to hold attribute values from a NetCDF file
! (read on PEO and broadcast to all PEs).
TYPE nc_fileinfo
  SEQUENCE
  REAL    :: att_value(max_att_len)
  LOGICAL :: l_exist
END TYPE nc_fileinfo

TYPE (nc_fileinfo) :: file_info

INTEGER (KIND=jpim), PARAMETER :: zhook_in  = 0
INTEGER (KIND=jpim), PARAMETER :: zhook_out = 1
REAL    (KIND=jprb)            :: zhook_handle

CHARACTER(LEN=*), PARAMETER :: RoutineName='ML_GET_VAR_ATT_REAL'

!  End of header
   
IF (lhook) CALL dr_hook(ModuleName//':'//RoutineName, zhook_in, zhook_handle)

! Obtain the communication group for this run
CALL gc_get_communicator(my_comm, icode)

! Obtain an object representing the multi-datatype structure which is
! going to be broadcast (see futher details above)
CALL setup_nml_type(no_of_types, mpl_nml_type, n_real_in = n_real, &
                    n_log_in = n_log)

! PE0 reads attribute values and adds them to the derived type file_info
IF ( mype == 0 ) THEN
  ! Get variable id
  nfid = fileid
  IF ( TRIM(var_name) == 'GLOBAL'   .OR.  TRIM(var_name) == 'global' .OR. &
       TRIM(var_name) == 'Global' ) THEN
    varid32 = NF90_GLOBAL
  ELSE
    nstatus = nf90_inq_varid (nfid, var_name, varid32)
    CALL nd_error (nstatus, 'ML_GET_VAR_ATT_REAL: INQ Varid', var_name)
  END IF

  ! Get values of attribute
  rval32(:) = 0.0
  nstatus   = nf90_get_att (nfid, varid32, att_name, rval32)

  ! If argument att_exists is present that means that the attribute is not
  ! compulsory. Therefore, if attribute found in the previous call to 
  ! nf90_get_att then change the local variable l_exist to True; otherwise
  ! leave it as False but there is no need to flag error.
  l_exist = .FALSE.
  IF ( PRESENT(att_exists)  ) THEN
    IF (nstatus == nf90_noerr) l_exist = .TRUE.  ! attrib exists

  ! If argument att_exists not present then the attribute is needed. Therefore
  ! call nd_error, which will stop the model only if the previous call to
  ! nf90_get_att reported error (as given by nstatus).
  ELSE
    CALL nd_error (nstatus, 'ML_GET_VAR_ATT_REAL: Get Attribute for ' //  &
                   TRIM(var_name), att_name)
  END IF

  ! Copy local variables into broadcast type
  file_info % l_exist           = l_exist
  file_info % att_value(1:ilen) = rval32(1:ilen)

END IF

! Distribute data to all other PEs
icode = 0
CALL mpl_bcast (file_info, 1, mpl_nml_type, 0, my_comm, icode)
IF ( icode /= 0 ) THEN
  ecode = fileid
  CALL ereport ('ML_GET_VAR_ATT_REAL', ecode, 'NML Broadcast Error')
END IF

! Operation on all PEs different from PE0: Transfer data from
! broadcast type to local variables.
IF ( mype /= 0 ) THEN
  l_exist        = file_info % l_exist
  rval32(1:ilen) = file_info % att_value(1:ilen)
END IF

! Operation on all PEs: Transfer data from local variables to
! OUT and (if present) optional INOUT arguments.
rvalue = rval32
IF ( PRESENT(att_exists) ) att_exists = l_exist

! Reset the multi-datatype structure allocated for broadcasting
CALL mpl_type_free (mpl_nml_type, icode)

IF (lhook) CALL dr_hook(ModuleName//':'//RoutineName, zhook_out, zhook_handle)

RETURN
END SUBROUTINE ml_get_var_att_real


! ------------------------------------------------------------------------
! Description:
!   Close NetCDF emission file.
!
! Method:
!   Call NF90_CLOSE to close a NetCDF file and return nstatus. Also call
!   ND_ERROR, which will stop the model if nstatus reports errors. Note
!   that this is done only by PE0.
! ------------------------------------------------------------------------
SUBROUTINE ml_fclose (fileid)

IMPLICIT NONE

! Subroutine arguments
INTEGER, INTENT(IN) :: fileid

! Local variables
INTEGER (KIND=integer32)       :: nstatus        ! netCDF return code

INTEGER (KIND=jpim), PARAMETER :: zhook_in  = 0
INTEGER (KIND=jpim), PARAMETER :: zhook_out = 1
REAL    (KIND=jprb)            :: zhook_handle

CHARACTER(LEN=*), PARAMETER :: RoutineName='ML_FCLOSE'

! End of header

IF (lhook) CALL dr_hook(ModuleName//':'//RoutineName, zhook_in, zhook_handle)

! Close file only by PE0
IF ( mype == 0 ) THEN
  nfid    = fileid
  nstatus = nf90_close(nfid)
  CALL nd_error (nstatus, 'ML_FCLOSE', ' ')
END IF

IF (lhook) CALL dr_hook(ModuleName//':'//RoutineName, zhook_out, zhook_handle)

RETURN
END SUBROUTINE ml_fclose

! -------------------------------------------------------------------------
! Description:
!   Checks return status of netcdf call and stop if needed.
!
! Method:
!   If the return status (nstatus) indicates error then:
!    * Close the NetCDF file by using NF90_CLOSE.
!    * Obtain error message from NF90_STRERROR and concatenate it with
!      the string cmessage, which contains more detailed information passed
!      from the routine where the error was found. Then print out
!      those messages and call EREPORT to stop the model.
! -------------------------------------------------------------------------
SUBROUTINE nd_error (nstatus, routine_name, cmessage)

IMPLICIT NONE

! Subroutine arguments
INTEGER (KIND=integer32), INTENT(IN) :: nstatus  ! netCDF return code

CHARACTER (LEN=*), INTENT(IN) :: routine_name ! routine where error found
CHARACTER (LEN=*), INTENT(IN) :: cmessage     ! error messg from that routine

! Local variables
INTEGER (KIND=integer32)       :: ncode

INTEGER (KIND=jpim), PARAMETER :: zhook_in  = 0
INTEGER (KIND=jpim), PARAMETER :: zhook_out = 1
REAL    (KIND=jprb)            :: zhook_handle

CHARACTER(LEN=*), PARAMETER :: RoutineName='ND_ERROR'

! End of header

IF (lhook) CALL dr_hook(ModuleName//':'//RoutineName, zhook_in, zhook_handle)

IF ( nstatus /= nf90_noerr ) THEN
  ! Close file only by PE0
  IF ( mype == 0 ) THEN
    ncode = nf90_close(nfid)
  END IF
  ecode = 1

  ! Printing error messages first to avoid passing
  ! a very long string argument to ereport
  WRITE (umMessage,'(a)') 'Error in ' // routine_name    //  ": "  // &
                  TRIM (nf90_strerror(nstatus))  //  " - " // &
                  TRIM (cmessage)
  CALL umPrint(umMessage,src='nd_error')
  CALL ereport (routine_name, ecode, 'NetCDF error')
END IF

IF (lhook) CALL dr_hook(ModuleName//':'//RoutineName, zhook_out, zhook_handle)

RETURN

END SUBROUTINE nd_error
! -------------------------------------------------------

END MODULE ukca_mach_learn_mod

