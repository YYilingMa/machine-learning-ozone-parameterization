!
! mo_art_mloz
! This module Calculate an ozone field from a set of coefficients 
! provided by input file
!
!
! ICON
!
! ---------------------------------------------------------------
! Copyright (C) 2004-2024, DWD, MPI-M, DKRZ, KIT, ETH, MeteoSwiss
! Contact information: icon-model.org
!
! See AUTHORS.TXT for a list of authors
! See LICENSES/ for license information
! SPDX-License-Identifier: BSD-3-Clause
! ---------------------------------------------------------------

MODULE mo_art_mloz
  ! ICON
  USE mo_kind,                ONLY: wp
  USE mo_math_constants,      ONLY: rad2deg
  USE mtime,                  ONLY: datetime 
  USE mo_model_domain,        ONLY: p_patch
  USE mo_exception,           ONLY: message, message_text, finish
  USE mo_parallel_config,     ONLY: nproma
  USE mo_nh_vert_interp,      ONLY: start_idx_threshold, prepare_lin_intp, prepare_extrap, &
                                  & prepare_cubic_intp !, prepare_extrap_ifspp
  USE mo_nonhydro_state,      ONLY: p_nh_state
  USE mo_upatmo_config,       ONLY: upatmo_config
  ! USE mo_nh_vert_extrap_utils,ONLY: t_expol_state
  USE mo_initicon_config,     ONLY: zpbl1, zpbl2, l_coarse2fine_mode
  USE mo_io_config,           ONLY: itype_pres_msl
  USE mo_run_config,          ONLY: num_lev
  ! USE mo_physical_constants,  ONLY: dtdz_standardatm !-6.5e-3[ K/m ]
  USE mo_fortran_tools,       ONLY: minval_1d, set_acc_host_or_device
  USE mo_physical_constants,  ONLY: amd, amo3, amw
  USE mo_o3_util,             ONLY: o3_timeint

  ! ART
  USE mo_art_chem_types_param, ONLY: t_chem_meta_mloz
  USE mo_art_data,             ONLY: p_art_data
  USE mo_art_chem_data,        ONLY: t_art_chem,  &
                                  &  t_art_mloz
  USE mo_art_atmo_data,        ONLY: t_art_atmo ! mo_art_atmo_data initialize variables
  USE mo_art_impl_constants,   ONLY: IART_QV

  IMPLICIT NONE
  PRIVATE
  PUBLIC   :: art_calc_mloz, lin_intp, temperature_intp

  ! Local variables
  !

  CONTAINS

  ! ----------------------------------
  ! --- Machine Learning-based parameterized Ozone (Nowack et al. 2018)
  ! --- by Yiling Ma/IMK-ASF 13.07.2024
  ! ----------------------------------
  SUBROUTINE art_calc_mloz(jg,jb,jcs,jce,nblks,nlev,nlev_ext,ncoefs,tracer,current_date)
  !<
  ! SUBROUTINE art_calc_mloz
  ! Routine to calculate the ozone concentration based on machine learning 
  ! based on Nowack et al. 2018
  ! Part of Module: mo_art_mloz
  ! Author: Yiling Ma, KIT
  !>

    INTEGER, INTENT(in) :: &
      &  jg,               &  !< patch id
      &  jb,jcs,jce,       &  !< loop indices
      &  nblks,nlev,nlev_ext, & ! dimension parameters. nlev:number of full levels
      &  ncoefs
    TYPE(t_chem_meta_mloz), INTENT(inout) ::  &
      &  tracer       !< structure for tracer (ozone); (nproma,nlev,nblks); number concentration
    TYPE(datetime), POINTER, INTENT(IN)  ::  &  
      &  current_date !< current simulation date
    ! local variables
    INTEGER :: jk, jc ! temporarily set to nlev_ext in MODULE mo_art_chem_init_meta 
    TYPE(t_art_atmo),POINTER    :: &
      &  art_atmo                   !< Pointer to ART atmo fields; Declaring a Pointer Variable
    TYPE(t_art_chem),POINTER    :: &
      &  art_chem                   !< Pointer to ART chem fields
    TYPE(t_art_mloz),POINTER    :: &
      &  art_mloz                   !< Pointer to machine learning coefs & ozone scalings

    ! INTEGER, ALLOCATABLE:: current_day(:)  ! not sure if different blocks share memory
    ! INTEGER, ALLOCATABLE,SAVE :: n_times(:)
    ! LOGICAL, ALLOCATABLE:: l_first(:)
    REAL(wp), ALLOCATABLE,SAVE :: daily_mean_temp(:,:,:)
    REAL(wp), ALLOCATABLE,SAVE :: daily_mean_feature(:,:,:) ! humidity: mmr - [kg/kg]
    REAL(wp), DIMENSION(nproma,nlev_ext) :: daily_mean_temp_extlev ! daily mean tem on external height coordinate
    REAL(wp), DIMENSION(nproma,nlev_ext) :: daily_mean_feature_extlev ! humidity: mmr - [kg/kg]
    INTEGER, ALLOCATABLE,SAVE :: current_day(:) ! different blocks share memory, but they are on the same day ! current day on each blk
    INTEGER, ALLOCATABLE,SAVE :: n_times(:) ! total model time steps in current day on each blk
    LOGICAL,SAVE :: l_first=.TRUE.
    INTEGER :: i,j,Crr_mon,Crr_day
    REAL(wp), DIMENSION(ncoefs) :: tvec ! single column temp
    REAL(wp), DIMENSION(nlev_ext) :: ml_ozone_pred ! single column ozone
    REAL(wp), DIMENSION(nproma,nlev_ext) :: ozone_pred ! ozone predicted by ML
    REAL(wp), DIMENSION(nproma,nlev) :: ozone_pred_nlev ! ML ozone interpolated to ICON level
    REAL(wp), DIMENSION(nlev_ext,ncoefs) :: mlmat ! ml_coeffs(1:nlev_ext,1:ncoefs)
    REAL(wp), DIMENSION(nproma,nlev) :: o3_meso_clim ! ozone climatology to replace mesosphere
    REAL(wp), DIMENSION(1:nproma,1:nlev,0:13) :: ml_o3_meso1 ! ozone climatology to replace mesosphere on 12 month
    ! minimum ozone value allowed to trap negatives etc.
    REAL, PARAMETER :: ozone_min_val = 1.0e-15
    REAL,PARAMETER :: ml_limit_ht = 50000. ! 50km/60km/75km
    CHARACTER(len=*), PARAMETER :: routine = 'mo_art_mloz'

    !--------------------------------------------
    ! for vertical interpolation
    !--------------------------------------------
    ! weighting factors and indices
    REAL(wp),DIMENSION(nproma,nlev,nblks) :: z_mc ! height coordinate of ICON full model levels
    REAL(wp),ALLOCATABLE,SAVE :: wfac_o3(:,:,:)
    REAL(wp),ALLOCATABLE,SAVE :: wfac_tem(:,:,:) ! weighting factor of upper level for linear interpolation, (nproma,nlev_ext,nblks)
    REAL(wp),ALLOCATABLE,SAVE :: & wfacpbl1_o3(:,:), wfacpbl2_o3(:,:), &
                      & wfacpbl1_tem(:,:), wfacpbl2_tem(:,:) ! weighting coefficient, (nproma,nblks)
    INTEGER,ALLOCATABLE,SAVE :: idx0_o3(:,:,:)    ! index of upper level, (nproma,nlev,nblks)
    INTEGER,ALLOCATABLE,SAVE :: idx0_tem(:,:,:), idx0_cub_tem(:,:,:) ! index of upper level, (nproma,nlev_ext,nblks)
    INTEGER,ALLOCATABLE,SAVE ::  kpbl1_tem(:,:), kpbl2_tem(:,:), kpbl1_o3(:,:), kpbl2_o3(:,:), & !(nproma,nblks)
                      & bot_idx_tem(:,:), bot_idx_o3(:,:), bot_idx_cub_tem(:,:) ! index of lowest level for which interpolation is possible (as opposed to extrapolation)
    ! Auxiliary fields for coefficients and filled input height fields
    REAL(wp),ALLOCATABLE,SAVE :: coef1_tem(:,:,:), coef2_tem(:,:,:), coef3_tem(:,:,:) !(nproma,nlev_ext,nblks)
    LOGICAL,DIMENSION(nproma,nlev) :: level_above_ht
    LOGICAL :: l_vert_intp=.TRUE. ! .TRUE.: excecute vertical interpolation to fit into external model height coordinate

    art_atmo => p_art_data(jg)%atmo
    art_chem => p_art_data(jg)%chem
    art_mloz => p_art_data(jg)%chem%param%mloz
    ! nblks = art_atmo%i_endblk - art_atmo%i_startblk ! number of blocks
    ! nproma = art_atmo%nproma ! length of each block
    z_mc = p_nh_state(jg)%metrics%z_mc
    ! npromz = art_atmo%npromz

    ! On the first timestep set the value of current_day
    ! cannot do ML ozone on first timestep as don't have daily
    ! mean temperature to work with.
    ! Initialise daily_mean_temp.
    ! Get ML coeffs and scalings. Only need to do on first timestep as these
    ! do not change during the run.

    IF (l_first) THEN ! initialization

      IF (.NOT. ALLOCATED(current_day)) THEN
        ALLOCATE(current_day(1:nblks))
        current_day(:) = current_date%date%day ! current_date is the day of month!
      END IF
      IF (.NOT. ALLOCATED(n_times)) THEN
        ALLOCATE(n_times(1:nblks))
        n_times(:) = 0
      END IF
      IF (.NOT. ALLOCATED(daily_mean_temp)) THEN
        ALLOCATE(daily_mean_temp(1:nproma, 1:nlev, 1:nblks))
        daily_mean_temp(:,:,:) = 0.0
      END IF
      IF (art_mloz%l_add_feature) THEN
        IF (.NOT. ALLOCATED(daily_mean_feature)) THEN
          ALLOCATE(daily_mean_feature(1:nproma, 1:nlev, 1:nblks))
          daily_mean_feature(:,:,:) = 0.0
        END IF
      END IF
      IF (l_vert_intp) THEN
        IF (.NOT. ALLOCATED(wfac_o3)) THEN
          ALLOCATE(wfac_o3(1:nproma, 1:nlev, 1:nblks))
          wfac_o3(:,:,:) = 0.0
        END IF
        IF (.NOT. ALLOCATED(wfac_tem)) THEN
          ALLOCATE(wfac_tem(1:nproma, 1:nlev, 1:nblks))
          wfac_tem(:,:,:) = 0.0
        END IF
        IF (.NOT. ALLOCATED(wfacpbl1_o3)) THEN
          ALLOCATE(wfacpbl1_o3(1:nproma, 1:nblks))
          wfacpbl1_o3(:,:) = 0.0
        END IF
        IF (.NOT. ALLOCATED(wfacpbl2_o3)) THEN
          ALLOCATE(wfacpbl2_o3(1:nproma, 1:nblks))
          wfacpbl2_o3(:,:) = 0.0
        END IF
        IF (.NOT. ALLOCATED(wfacpbl1_tem)) THEN
          ALLOCATE(wfacpbl1_tem(1:nproma, 1:nblks))
          wfacpbl1_tem(:,:) = 0.0
        END IF
        IF (.NOT. ALLOCATED(wfacpbl2_tem)) THEN
          ALLOCATE(wfacpbl2_tem(1:nproma, 1:nblks))
          wfacpbl2_tem(:,:) = 0.0
        END IF
        IF (.NOT. ALLOCATED(idx0_o3)) THEN
          ALLOCATE(idx0_o3(1:nproma, 1:nlev, 1:nblks))
          idx0_o3(:,:,:) = 0
        END IF
        IF (.NOT. ALLOCATED(idx0_tem)) THEN
          ALLOCATE(idx0_tem(1:nproma, 1:nlev_ext, 1:nblks))
          idx0_tem(:,:,:) = 0
        END IF
        IF (.NOT. ALLOCATED(idx0_cub_tem)) THEN
          ALLOCATE(idx0_cub_tem(1:nproma, 1:nlev_ext, 1:nblks))
          idx0_cub_tem(:,:,:) = 0
        END IF
        IF (.NOT. ALLOCATED(kpbl1_tem)) THEN
          ALLOCATE(kpbl1_tem(1:nproma, 1:nblks))
          kpbl1_tem(:,:) = 0
        END IF
        IF (.NOT. ALLOCATED(kpbl2_tem)) THEN
          ALLOCATE(kpbl2_tem(1:nproma, 1:nblks))
          kpbl2_tem(:,:) = 0
        END IF
        IF (.NOT. ALLOCATED(kpbl1_o3)) THEN
          ALLOCATE(kpbl1_o3(1:nproma, 1:nblks))
          kpbl1_o3(:,:) = 0
        END IF
        IF (.NOT. ALLOCATED(kpbl2_o3)) THEN
          ALLOCATE(kpbl2_o3(1:nproma, 1:nblks))
          kpbl2_o3(:,:) = 0
        END IF
        IF (.NOT. ALLOCATED(bot_idx_tem)) THEN
          ALLOCATE(bot_idx_tem(1:nproma, 1:nblks))
          bot_idx_tem(:,:) = 0
        END IF
        IF (.NOT. ALLOCATED(bot_idx_o3)) THEN
          ALLOCATE(bot_idx_o3(1:nproma, 1:nblks))
          bot_idx_o3(:,:) = 0
        END IF
        IF (.NOT. ALLOCATED(bot_idx_cub_tem)) THEN
          ALLOCATE(bot_idx_cub_tem(1:nproma, 1:nblks))
          bot_idx_cub_tem(:,:) = 0
        END IF
        IF (.NOT. ALLOCATED(coef1_tem)) THEN
          ALLOCATE(coef1_tem(1:nproma, 1:nlev_ext, 1:nblks))
          coef1_tem(:,:,:) = 0.0
        END IF
        IF (.NOT. ALLOCATED(coef2_tem)) THEN
          ALLOCATE(coef2_tem(1:nproma, 1:nlev_ext, 1:nblks))
          coef2_tem(:,:,:) = 0.0
        END IF
        IF (.NOT. ALLOCATED(coef3_tem)) THEN
          ALLOCATE(coef3_tem(1:nproma, 1:nlev_ext, 1:nblks))
          coef3_tem(:,:,:) = 0.0
        END IF
        
        !-------------------------------------------
        ! prepare weighting factor for vertical interp
        ! implement only once on the first timestep
        !-------------------------------------------
        !!! o3 ukesm -> icon (full levels)
        ! It is assumed that the highest level of the input data is at least
        ! as high as the highest level of the output data
        ! otherwise perform extrapolation or using constant values (l_extrap)
        CALL prepare_lin_intp(art_mloz%z3d_ext(:,1:nlev_ext,:), z_mc, &
              &                  nblks, jce-jcs+1,       &
              &                  nlev_ext, nlev,         &
              &                  wfac_o3, idx0_o3, bot_idx_o3, lextrap=.TRUE., lacc=.FALSE.)
        ! Computes coefficient fields for vertical extrapolation below 
        ! the surface level of the input model
        CALL prepare_extrap(art_mloz%z3d_ext(:,1:nlev_ext,:),               &
                          &  nblks, jce-jcs+1, nlev_ext,    &
                          &  kpbl1_o3, wfacpbl1_o3,         &
                          &  kpbl2_o3, wfacpbl2_o3, lacc=.FALSE.) 

        !!! temp icon -> ukesm
        CALL prepare_lin_intp(z_mc, art_mloz%z3d_ext(:,1:nlev_ext,:),               &
              &                  nblks, jce-jcs+1,            &
              &                  nlev, nlev_ext,  &
              &                  wfac_tem, idx0_tem, bot_idx_tem, lextrap=.TRUE.,lacc=.FALSE.)
        !! Computes coefficient fields for vertical extrapolation below 
        !! the surface level of the input model
        CALL prepare_extrap(z_mc,                                    &
                          &  nblks, jce-jcs+1, nlev, &
                          &  kpbl1_tem, wfacpbl1_tem, &
                          &  kpbl2_tem, wfacpbl2_tem, lacc=.FALSE.) 
        
        !! Computes coefficient fields for cubic vertical interpolation
        !! It is assumed that the highest level of the input data is at least
        !! as high as the highest level of the output data
        CALL prepare_cubic_intp(z_mc, art_mloz%z3d_ext(:,1:nlev_ext,:),   &
                                nblks, jce-jcs+1, nlev, nlev_ext, &
                                coef1_tem, coef2_tem, coef3_tem, &
                                idx0_cub_tem, bot_idx_cub_tem, lacc=.FALSE.)
      END IF

      l_first                = .FALSE.

    END IF

    IF (.NOT. l_first) THEN

      Crr_mon = current_date%date%month
      Crr_day = current_date%date%day

      ! WRITE(message_text,*) 'art_mloz%current_day(jb)=',art_mloz%current_day(jb), &
      !       & ', current_date%date%day=',current_date%date%day
      ! DO ML CALCULATION IF THE DAY NUMBER CHANGES
      IF (current_day(jb) /= Crr_day ) THEN 

        WRITE(message_text,*) 'jb=', jb,'current_date%date%month=',current_date%date%month,', Crr_day=',Crr_day
        CALL message(TRIM(routine),TRIM(message_text))

        ! Linearly interpolate the ozone climatology to replace the mesospheric ozone
        ! art_mloz%ml_o3_meso(:,:,jb,Crr_mon) ! no interpolation
        ml_o3_meso1(:,:,1:12) = art_mloz%ml_o3_meso(:,:,jb,1:12)
        ml_o3_meso1(:,:,0) = art_mloz%ml_o3_meso(:,:,jb,12)
        ml_o3_meso1(:,:,13) = art_mloz%ml_o3_meso(:,:,jb,1)

        CALL o3_timeint(jcs = jcs, jce = jce, kbdim = nproma,       &
               &          nlev_pres    = nlev,                      &
               &          ext_o3       = ml_o3_meso1(:,:,:),        &
               &          current_date = current_date,              &
               &          o3_time_int  = o3_meso_clim(:,:)          ) 
        
        ! calculate daily mean temperature; standardize daily_mean_temp
        daily_mean_temp(:,:,jb) = daily_mean_temp(:,:,jb)/n_times(jb) 
        daily_mean_temp(:,:,jb) = (daily_mean_temp(:,:,jb)-art_mloz%ml_temp_mean(:,:,jb))/art_mloz%ml_temp_scale(:,:,jb)

        !!!
        ! vertically interpolate temp from icon to ukesm height
        IF (l_vert_intp) CALL temperature_intp(                               &
                          &  daily_mean_temp(:,:,jb), daily_mean_temp_extlev(:,:), &
                          &  z_mc(:,:,jb), art_mloz%z3d_ext(:,1:nlev_ext,jb), &
                          &  nblks, jcs, jce, jb,                             &
                          &  nlev, nlev_ext,                                  &
                          &  coef1_tem(:,:,jb), coef2_tem(:,:,jb),            &
                          &  coef3_tem(:,:,jb), wfac_tem(:,:,jb),             & 
                          &  idx0_cub_tem(:,:,jb), idx0_tem(:,:,jb),          &
                          &  bot_idx_cub_tem(:,jb), bot_idx_tem(:,jb),        &
                          &  wfacpbl1_tem(:,jb), kpbl1_tem(:,jb),             &
                          &  wfacpbl2_tem(:,jb), kpbl2_tem(:,jb),             &
                          &  l_restore_sfcinv=.FALSE., l_hires_corr=.FALSE.,  &
                          &  l_sfc_inv=.FALSE.,                               &
                          &  lacc=.FALSE., extrapol_dist=-1500._wp)
        !! I think l_sfc_inv&l_restore_sfcinv should be on and off together.

        ! l_hires_corr=.TRUE.: Reduce the surface inversion strength depending on the slope 
        ! of the target point and on the height difference between the source and target topography.
        ! designed for the case that the target model has a significantly
        ! finer spatial resolution than the source model, implying that nocturnal
        ! surface inversions should be removed on grid points lying higher than on
        ! the source grid (i.e. mountain ranges not resolved in the source model)
        ! and on slope points because cold air drainage in reality lets the cold air
        ! accumulate at the valley bottom.

        ! WRITE(message_text,*) 'jb=', jb, ', AVERAGE(daily_mean_temp(:,:,jb))=', &
        ! & SUM(daily_mean_temp(:,:,jb))/SIZE(daily_mean_temp(:,:,jb))
        ! CALL message(TRIM(routine),TRIM(message_text))
        
        ! WRITE(message_text,*) 'jb=', jb, ', AVERAGE(daily_mean_temp_extlev=', &
        ! & SUM(daily_mean_temp_extlev)/SIZE(daily_mean_temp_extlev)
        ! CALL message(TRIM(routine),TRIM(message_text))
        ! output daily_mean_temp_extlev as ncfile
        ! p_nh_state(jg)%diag%extra_3d_ptr(1)%p_3d(:,nlev_ext+1:nlev,jb) = 0._wp
        ! p_nh_state(jg)%diag%extra_3d_ptr(1)%p_3d(:,1:nlev_ext,jb) = daily_mean_temp_extlev(:,:)

        IF (art_mloz%l_add_feature) THEN

          ! calculate daily mean humidity; standardize daily_mean_feature
          daily_mean_feature(:,:,jb) = daily_mean_feature(:,:,jb)/n_times(jb) 

          ! WRITE(message_text,*) 'jb=', jb, ', AVERAGE(daily_mean_feature(:,:,jb)=', &
          ! & SUM(daily_mean_feature(:,:,jb))/SIZE(daily_mean_feature(:,:,jb))
          ! CALL message(TRIM(routine),TRIM(message_text))

          daily_mean_feature(:,:,jb) = (daily_mean_feature(:,:,jb)-art_mloz%ml_feature_mean(:,:,jb))/art_mloz%ml_feature_scale(:,:,jb)
          ! daily_mean_feature(:,:,jb) = art_mloz%ml_feature_mean(:,:,jb)
          ! ml_feature_mean and ml_feature_scale is mmr (for humidity) or Pa (for pressure)

          ! perform interpolation on specific humidity; temp & pressure on input&ouput grid is needed
          !! hum here was standardized
          IF (art_mloz%feature_added == "h") CALL temperature_intp(             &
                            &  daily_mean_feature(:,:,jb), daily_mean_feature_extlev(:,:), &
                            &  z_mc(:,:,jb), art_mloz%z3d_ext(:,1:nlev_ext,jb), &
                            &  nblks, jcs, jce, jb,                             &
                            &  nlev, nlev_ext,                                  &
                            &  coef1_tem(:,:,jb), coef2_tem(:,:,jb),            &
                            &  coef3_tem(:,:,jb), wfac_tem(:,:,jb),             & 
                            &  idx0_cub_tem(:,:,jb), idx0_tem(:,:,jb),          &
                            &  bot_idx_cub_tem(:,jb), bot_idx_tem(:,jb),        &
                            &  wfacpbl1_tem(:,jb), kpbl1_tem(:,jb),             &
                            &  wfacpbl2_tem(:,jb), kpbl2_tem(:,jb),             &
                            &  l_restore_sfcinv=.FALSE., l_hires_corr=.FALSE.,  &
                            &  l_sfc_inv=.FALSE.,                               &
                            &  lacc=.FALSE., extrapol_dist=-1500._wp)
              ! l_hires_corr: applies corrections / limits for coarse-to-fine grid interpolation
              ! l_sfc_inv: if modified temperature with surface inversion removed below zpbl1

          IF (art_mloz%feature_added == "p") CALL lin_intp(                  & 
                            &         daily_mean_feature(:,:,jb), daily_mean_feature_extlev(:,:), &              
                            &         nblks, jcs, jce, jb,                      &
                            &         nlev,      nlev_ext,                      &
                            &         wfac_tem(:,:,jb), idx0_tem(:,:,jb),         &
                            &         bot_idx_tem(:,jb), wfacpbl1_tem(:,jb),      &
                            &         kpbl1_tem(:,jb), wfacpbl2_tem(:,jb),        &
                            &         kpbl2_tem(:,jb),                           &
                            &         l_loglin=.true., l_pd_limit=.false.,     &
                            &         l_extrapol=.true., lacc=.FALSE.          )
                          ! l_pd_limit: Positive definite limiter
                          ! l_extrapol: linear extrapolation, using the gradient between (by default) 
                          !             1000 m and 500 m AGL

          ! WRITE(message_text,*) 'jb=', jb, ', AVERAGE(daily_mean_feature(:,:,jb))=', &
          ! & SUM(daily_mean_feature(:,:,jb))/SIZE(daily_mean_feature(:,:,jb))
          ! CALL message(TRIM(routine),TRIM(message_text))
          
          ! WRITE(message_text,*) 'jb=', jb, ', AVERAGE(daily_mean_feature_extlev=', &
          ! & SUM(daily_mean_feature_extlev)/SIZE(daily_mean_feature_extlev)
          ! CALL message(TRIM(routine),TRIM(message_text))

          p_nh_state(jg)%diag%extra_3d_ptr(1)%p_3d(:,nlev_ext+1:nlev,jb) = 0._wp
          p_nh_state(jg)%diag%extra_3d_ptr(1)%p_3d(:,1:nlev_ext,jb) = daily_mean_feature_extlev(:,:)
          p_nh_state(jg)%diag%extra_3d_ptr(2)%p_3d(:,1:nlev,jb) = daily_mean_feature(:,:,jb)

        END IF

        DO i=jcs,jce

          IF (art_mloz%l_add_feature) THEN
            tvec(1:nlev_ext) = daily_mean_feature_extlev(i,:)
            tvec(nlev_ext+1:nlev_ext*2) = daily_mean_temp_extlev(i,:)
          ELSE
            tvec(1:nlev_ext) = daily_mean_temp_extlev(i,:)
          END IF
          !! tqv_vec(1:lev_ext) = daily_mean_temp_extlev(i,:)
          !! tqv_vec(lev_ext+1:lev_ext*2) = daily_mean_feature_extlev(i,:)
          mlmat(:,:) = art_mloz%ml_coefs(i,:,jb,:) ! ml_coefs(nproma,nlev_ext,nblks,ncoefs), mlmat(nlev_ext,n_coefs)

          ml_ozone_pred(:) = MATMUL(mlmat(:,:),tvec(:))
          IF (art_mloz%l_intercept) ml_ozone_pred(:) = ml_ozone_pred(:) + art_mloz%ml_intcs(i,:,jb) ! unit of intercept doesn't matter

          ! p_nh_state(jg)%diag%extra_3d_ptr(3)%p_3d(i,nlev_ext+1:nlev,jb) = 0._wp
          ! p_nh_state(jg)%diag%extra_3d_ptr(3)%p_3d(i,1:nlev_ext,jb) = mlmat(7,1:nlev_ext)
          ! p_nh_state(jg)%diag%extra_3d_ptr(4)%p_3d(i,nlev_ext+1:nlev,jb) = 0._wp
          ! p_nh_state(jg)%diag%extra_3d_ptr(4)%p_3d(i,1:nlev_ext,jb) = mlmat(7,nlev_ext+1:nlev_ext*2)

          ! WRITE(message_text,*) 'jb=', jb, 'AVERAGE(ml_ozone_pred(:))=', &
          ! & SUM(ml_ozone_pred(:))/SIZE(ml_ozone_pred(:))
          ! CALL message(TRIM(routine),TRIM(message_text))

          ozone_pred(i,:) = ml_ozone_pred(:)

        END DO

        ! WRITE(message_text,*) 'jb=', jb, 'AVERAGE(art_mloz%ml_ozone_scale(:,:,jb))=', &
        ! & SUM(art_mloz%ml_ozone_scale(:,:,jb))/SIZE(art_mloz%ml_ozone_scale(:,:,jb))
        ! CALL message(TRIM(routine),TRIM(message_text))

        ! WRITE(message_text,*) 'jb=', jb, 'AVERAGE(art_mloz%ml_ozone_mean(:,:,jb))=', &
        ! & SUM(art_mloz%ml_ozone_mean(:,:,jb))/SIZE(art_mloz%ml_ozone_mean(:,:,jb))
        ! CALL message(TRIM(routine),TRIM(message_text))

        ! output ozone_pred as ncfile
        ! p_nh_state(jg)%diag%extra_3d_ptr(5)%p_3d(:,nlev_ext+1:nlev,jb) = 0._wp
        ! p_nh_state(jg)%diag%extra_3d_ptr(5)%p_3d(:,1:nlev_ext,jb) = ozone_pred(:,:)
        ! p_nh_state(jg)%diag%extra_3d_ptr(6)%p_3d(:,1:nlev,jb) = ozone_pred_nlev(:,:)


        IF (art_mloz%l_yscale_130lev) THEN

          CALL lin_intp(ozone_pred, ozone_pred_nlev,         &               
              &         nblks, jcs, jce, jb,                 &
              &         nlev_ext,      nlev,                 &
              &         wfac_o3(:,:,jb), idx0_o3(:,:,jb),    &
              &         bot_idx_o3(:,jb), wfacpbl1_o3(:,jb), &
              &         kpbl1_o3(:,jb), wfacpbl2_o3(:,jb),   &
              &         kpbl2_o3(:,jb),                      &
              &         l_loglin=.false., l_pd_limit=.false., &
              &         l_extrapol=.false., lacc=.FALSE.     )
            ! l_pd_limit: Positive definite limiter
            ! l_extrapol: linear extrapolation, using the gradient between (by default) 
            !             1000 m and 500 m AGL

          level_above_ht = (z_mc(:,:,jb) > ml_limit_ht) ! for 130lev ozone scalings
          ! rescale using provided coefs
          WHERE(.NOT. level_above_ht) ozone_pred_nlev(:,:) =                   &
                      & ozone_pred_nlev(:,:) * art_mloz%ml_ozone_scale(:,:,jb) &
                      & + art_mloz%ml_ozone_mean(:,:,jb)

          ! replace the ozone above ml_limit_ht with climatologies
          WHERE(level_above_ht) ozone_pred_nlev(:,:) = o3_meso_clim(:,:) * amd/amo3 
          ! replace the top 3 lev with climatologies in order to prevent lookup table overflow
          ! ozone_pred_nlev(:,1:3) = art_mloz%ml_o3_meso(:,1:3,jb,Crr_mon) * amd/amo3

          ! IF (art_mloz%nlev_ext < 76) THEN
          !   ! replace the lowest 6 levels with climatologies
          !   ozone_pred_nlev(:,nlev-5:nlev) = art_mloz%ml_o3_meso(:,nlev-5:nlev,jb)*amd/amo3 
          ! END IF
          IF (nlev_ext < art_mloz%n_zfile) THEN
            ! replace the lowest 3 levels with climatologies
            ozone_pred_nlev(:,nlev-(art_mloz%n_zfile-nlev_ext-1):nlev) = o3_meso_clim(:,nlev-(art_mloz%n_zfile-nlev_ext-1):nlev)*amd/amo3 
          END IF

          ! convert from volume mixing ratio to number concentration
          ! ml_ozone_scale & ml_ozone_mean are both vmr, so ozone_pred_nlev is already vmr
          tracer%tracer(:,:,jb) = ozone_pred_nlev(:,:) * art_chem%vmr2Nconc(:,:,jb) ! tracer%tracer should be number concentration 

        ELSE ! use ukesm o3 scalings with nlev_ext

          ! rescale using provided ukesm o3 scalings
          ozone_pred(:,:) = ozone_pred(:,:) * art_mloz%ml_ozone_scale(:,:,jb) &
                            & + art_mloz%ml_ozone_mean(:,:,jb)
          ! output ozone_pred as ncfile
          ! p_nh_state(jg)%diag%extra_3d_ptr(2)%p_3d(:,nlev_ext+1:nlev,jb) = 0._wp
          ! p_nh_state(jg)%diag%extra_3d_ptr(2)%p_3d(:,1:nlev_ext,jb) = ozone_pred(:,:)

          ! vertically interpolate o3 from ukesm to icon model height
          ! ozone_pred_nlev(:,:) = ozone_min_val
          ! ozone_pred_nlev(:,15:nlev_ext+14) = ozone_pred(:,:)

          CALL lin_intp(ozone_pred, ozone_pred_nlev,         &               
              &         nblks, jcs, jce, jb,                 &
              &         nlev_ext,      nlev,                 &
              &         wfac_o3(:,:,jb), idx0_o3(:,:,jb),    &
              &         bot_idx_o3(:,jb), wfacpbl1_o3(:,jb), &
              &         kpbl1_o3(:,jb), wfacpbl2_o3(:,jb),   &
              &         kpbl2_o3(:,jb),                      &
              &         l_loglin=.false., l_pd_limit=.false., &
              &         l_extrapol=.false., lacc=.FALSE.     )
            ! l_pd_limit: Positive definite limiter
            ! l_extrapol: linear extrapolation, using the gradient between (by default) 
            !             1000 m and 500 m AGL

          level_above_ht = (z_mc(:,:,jb) > ml_limit_ht) ! for 76lev ozone scalings
          ! replace the ozone above ml_limit_ht with climatologies
          WHERE(level_above_ht) ozone_pred_nlev(:,:) = o3_meso_clim(:,:)
          ! replace the top 3 lev with climatologies in order to prevent lookup table overflow
          ! ozone_pred_nlev(:,1:3) = art_mloz%ml_o3_meso(:,1:3,jb,Crr_mon)
          ! replace the lowest 3/2 levels with climatologies
          ozone_pred_nlev(:,nlev-(art_mloz%n_zfile-nlev_ext-1):nlev) = o3_meso_clim(:,nlev-(art_mloz%n_zfile-nlev_ext-1):nlev)

          ! IF (nlev_ext < 76) THEN
          !   ! replace the lowest 6 levels with climatologies
          !   ozone_pred_nlev(:,nlev-5:nlev) = art_mloz%ml_o3_meso(:,nlev-5:nlev,jb) 
          ! END IF

          ! convert from mass mixing ratio to number concentration
          tracer%tracer(:,:,jb) = ozone_pred_nlev * amd/amo3 * art_chem%vmr2Nconc(:,:,jb) ! tracer%tracer should be number concentration

        END IF

        ! WRITE(message_text,*) 'jb=', jb, 'AVERAGE(ozone_pred_nlev(:,:))=', &
        ! & SUM(ozone_pred_nlev(:,:))/SIZE(ozone_pred_nlev(:,:))
        ! CALL message(TRIM(routine),TRIM(message_text))

        ! WRITE(message_text,*) 'level_above_ht(1,10:17)=', level_above_ht(1,10:17)
        ! CALL message(TRIM(routine),TRIM(message_text))


        ! reset minimum value
        WHERE(tracer%tracer(:,:,jb) < ozone_min_val) tracer%tracer(:,:,jb) = ozone_min_val
        
        ! WRITE(message_text,*) "jb=",jb,"AVERAGE(art_chem%water_tracers(:,1:60,jb,IART_QV)/art_chem%vmr2Nconc(:,1:60,jb))=", &
        ! & SUM(art_chem%water_tracers(:,1:60,jb,IART_QV)/art_chem%vmr2Nconc(:,1:60,jb))/SIZE(art_chem%water_tracers(:,1:60,jb,IART_QV))
        ! CALL message(TRIM(routine),TRIM(message_text))

        ! DEALLOCATE(tvec)
        ! DEALLOCATE(ml_ozone_pred)

        ! at end of ML calculation update value of current_day_number
        ! to enable calculation of daily mean
        current_day(jb) = Crr_day
        ! reset value of n_times & daily_mean_temp
        n_times(jb) = 0
        daily_mean_temp(:,:,jb) = 0.0
        IF (art_mloz%l_add_feature) daily_mean_feature(:,:,jb) = 0.0
        
      END IF

      ! if we are in the current day then start calculating daily mean temperature
      IF (current_day(jb) == Crr_day) THEN

        daily_mean_temp(:,:,jb) = daily_mean_temp(:,:,jb) + art_atmo%temp(:,:,jb) !
        ! WRITE(message_text,*) "jb=",jb,", AVERAGE(art_atmo%daily_mean_temp(:,:,jb))=", &
        !       & SUM(art_atmo%daily_mean_temp(:,:,jb))/SIZE(art_atmo%daily_mean_temp(:,:,jb))
        ! CALL message(message_text)

        IF (art_mloz%l_add_feature) THEN
          IF (art_mloz%feature_added == "h") daily_mean_feature(:,:,jb) = &
          & daily_mean_feature(:,:,jb) + art_chem%water_tracers(:,:,jb,IART_QV)*(amw/amd)/art_chem%vmr2Nconc(:,:,jb)

          IF (art_mloz%feature_added == "p") daily_mean_feature(:,:,jb) = &  
          & daily_mean_feature(:,:,jb) + art_atmo%pres(:,:,jb)
        END IF

        ! daily_mean_humidity is mmr; water_tracers is Nconc

        ! USE mo_run_config,                  ONLY: iqv
        ! tracer%tracer(:,:,jb,iqv)  tracer(:,:,jb,iqv)

        n_times(jb) = n_times(jb) + 1

      END IF
    END IF

  END SUBROUTINE art_calc_mloz


  !-------------
  !>
  !! SUBROUTINE lin_intp
  !! Performs linear vertical interpolation of a 3D field
  !!
  !! Required input fields: 3D input field to be interpolated,
  !! coefficient fields from prepare_lin_intp and prepare_extrap
  !! Output: interpolated 3D field
  !!
  !! Note: field values below the surface level of the input field are
  !! computed by linear extrapolation, using the gradient of the lowest (by default)
  !! 500 m above ground. Do not use for temperature!
  !!
  !! Setting l_loglin=.TRUE. activates logarithmic interpolation
  !!
  SUBROUTINE lin_intp(f3d_in, f3d_out,                          &
                      nblks, jcs, jce, jb, nlevs_in, nlevs_out, &
                      wfac, idx0, bot_idx, wfacpbl1, kpbl1,     &
                      wfacpbl2, kpbl2, l_loglin, l_pd_limit,    &
                      l_extrapol, lower_limit, lacc             )

    ! Atmospheric fields
    REAL(wp), INTENT(IN)  :: f3d_in (:,:) ! input field
    REAL(wp), INTENT(OUT) :: f3d_out(:,:) ! output field (on ICON vertical grid)

    ! Dimension parameters
    INTEGER , INTENT(IN) :: nblks      ! Number of blocks
    INTEGER , INTENT(IN) :: jcs,jce    ! Length of last block
    INTEGER , INTENT(IN) :: nlevs_in   ! Number of input levels
    INTEGER , INTENT(IN) :: nlevs_out  ! Number of output levels
    INTEGER , INTENT(IN) :: jb         ! index of current block

    ! Coefficients
    REAL(wp), INTENT(IN) :: wfac(:,:)    ! weighting factor of upper level
    INTEGER , INTENT(IN) :: idx0(:,:)    ! index of upper level
    INTEGER , INTENT(IN) :: bot_idx(:)   ! index of lowest level for which interpolation is possible
    INTEGER , INTENT(IN) :: kpbl1(:)     ! index of model level immediately above (by default) 500 m AGL
    REAL(wp), INTENT(IN) :: wfacpbl1(:)  ! corresponding interpolation coefficient
    INTEGER , INTENT(IN) :: kpbl2(:)     ! index of model level immediately above (by default) 1000 m AGL
    REAL(wp), INTENT(IN) :: wfacpbl2(:)  ! corresponding interpolation coefficient

    ! Control switches
    LOGICAL,  INTENT(IN) :: l_loglin    ! switch for logarithmic interpolation
    LOGICAL,  INTENT(IN) :: l_pd_limit  ! switch for use of positive definite limiter
    LOGICAL,  INTENT(IN) :: l_extrapol  ! switch for use of downward extrapolation (no-gradient condition otherwise)
    LOGICAL, INTENT(IN), OPTIONAL :: lacc ! if true use OpenACC

    REAL(wp), INTENT(IN), OPTIONAL :: lower_limit ! lower limit of variable

    ! LOCAL VARIABLES

    INTEGER  :: jk, jc
    REAL(wp) :: zf_in_tr(nproma,nlevs_in), zf_in_lim(nproma,nlevs_in), z_limit, f3d_z1, f3d_z2, vgrad_f3d
    LOGICAL :: lzacc ! non-optional version of lacc
    CHARACTER(len=*), PARAMETER :: routine = 'lin_intp'
!-------------------------------------------------------------------------

    CALL set_acc_host_or_device(lzacc, lacc)

    ! return, if nothing to do:
    IF ((nblks == 0) .OR. ((nblks == 1) .AND. ((jce-jcs+1) == 0))) RETURN

    IF (PRESENT(lower_limit)) THEN
      z_limit = lower_limit
    ELSE
      z_limit = -999._wp
    ENDIF

    ! WRITE(message_text,*) 'jb=', jb, 'bot_idx(:)=', bot_idx(:)
    ! CALL message(TRIM(routine),TRIM(message_text))
  

    !$ACC DATA IF(lzacc) &
    !$ACC   PRESENT(f3d_in, f3d_out, wfac, idx0, bot_idx, kpbl1, wfacpbl1, kpbl2, wfacpbl2) &
    !$ACC   CREATE(zf_in_tr, zf_in_lim)

!$OMP PARALLEL
!$OMP DO PRIVATE(jk,jc,zf_in_tr,zf_in_lim,f3d_z1,f3d_z2,vgrad_f3d) ICON_OMP_DEFAULT_SCHEDULE


    IF (jb == nblks) THEN
      !$ACC PARALLEL LOOP GANG VECTOR COLLAPSE(2) DEFAULT(PRESENT) ASYNC(1) IF(lzacc)
      DO jc = jce+1, nproma
        f3d_out(jc,:)  = -99.0_wp
      END DO
      !$ACC END PARALLEL LOOP
    ENDIF

    ! WRITE(*,*) 'jb=', jb, 'nblks=', &
    ! & nblks, 'nproma=', nproma, 'nlen(jb)=', nlen(jb)


    !$ACC PARALLEL DEFAULT(PRESENT) ASYNC(1) IF(lzacc)
    !$ACC LOOP SEQ
    DO jk = 1, nlevs_in
      !$ACC LOOP GANG(STATIC: 1) VECTOR
      DO jc = jcs, jce
        IF (l_pd_limit) THEN
          zf_in_lim(jc, jk) = MAX(z_limit,f3d_in(jc, jk))
        ELSE
          zf_in_lim(jc, jk) = f3d_in(jc, jk)
        ENDIF
        IF (l_loglin) THEN
          zf_in_tr(jc, jk) = LOG(MAX(1.e-20_wp,zf_in_lim(jc, jk)))
        ELSE
          zf_in_tr(jc, jk) = zf_in_lim(jc, jk)
        ENDIF
      ENDDO
    ENDDO

    !$ACC LOOP SEQ
    DO jk = 1, nlevs_out
      !$ACC LOOP GANG(STATIC: 1) VECTOR PRIVATE(f3d_z1, f3d_z2, vgrad_f3d)
      DO jc = jcs, jce
        IF (jk <= bot_idx(jc)) THEN ! if below the bot_idx level 

          ! linear interpolation
          f3d_out(jc,jk) = wfac(jc,jk)*zf_in_tr(jc,idx0(jc,jk)) + &
            (1._wp-wfac(jc,jk))*zf_in_tr(jc,idx0(jc,jk)+1)

          IF (l_loglin) f3d_out(jc,jk) = EXP(f3d_out(jc,jk))

        ELSE IF (l_extrapol) THEN

          ! linear extrapolation, using the gradient between (by default) 1000 m and 500 m AGL
          ! Logarithmic computation is not used here because it would be numerically unstable for extrapolation

          ! Field value at height zpbl1
          f3d_z1 = wfacpbl1(jc) *zf_in_lim(jc,kpbl1(jc)  ) +  &
            (1._wp-wfacpbl1(jc))*zf_in_lim(jc,kpbl1(jc)+1)

          ! Field value at height zpbl2
          f3d_z2 = wfacpbl2(jc) *zf_in_lim(jc,kpbl2(jc)) +  &
            (1._wp-wfacpbl2(jc))*zf_in_lim(jc,kpbl2(jc)+1)

          ! vertical gradient
          vgrad_f3d = (f3d_z2-f3d_z1)/(zpbl2-zpbl1)

          ! wfac carries the (negative) extrapolation distance (in m) in this case
          f3d_out(jc,jk) = zf_in_lim(jc,nlevs_in) + vgrad_f3d*wfac(jc,jk)

        ELSE ! use no-gradient condition for extrapolation

          f3d_out(jc,jk) = zf_in_lim(jc,nlevs_in)

        ENDIF
      ENDDO
    ENDDO

    IF (l_pd_limit) THEN
      !$ACC LOOP SEQ
      DO jk = 1, nlevs_out
        !$ACC LOOP GANG(STATIC: 1) VECTOR
        DO jc = jcs,jce
          f3d_out(jc, jk) = MAX(z_limit,f3d_out(jc, jk))
        ENDDO
      ENDDO
    ENDIF
      !$ACC END PARALLEL
    !$ACC WAIT
    !$ACC END DATA
!$OMP END DO NOWAIT
!$OMP END PARALLEL

  END SUBROUTINE lin_intp


  !-------------
  !>
  !! from SUBROUTINE temperature_intp in mo_nh_vert_interp.f90
  !! Performs vertical interpolation and extrapolation of temperature/specific humidity
  !!
  !! Required input fields: 3D input field to be interpolated,
  !! coefficient fields from prepare_lin/cubic_intp and prepare_extrap
  !! Output: interpolated 3D field
  !!
  !! Performs cubic interpolation where possible, turning to linear interpolation
  !! close to the surface
  !! The most important ingredient of the refined temperature extrapolation
  !! is to remove the surface inversion, if present, before the extrapolation
  !! and to add it again afterwards with a variable weighting coefficient,
  !! accounting for the slope of the target grid point and for the height
  !! difference between source and target grid
  !!
  !! shouldn't switch on surface inversion or l_hires_corr if variable is standardized! 
  SUBROUTINE temperature_intp(temp_in, temp_out, z3d_in, z3d_out,            &
                             nblks, jcs, jce, jb, nlevs_in, nlevs_out,       &
                             coef1, coef2, coef3, wfac_lin,                  &
                             idx0_cub, idx0_lin, bot_idx_cub, bot_idx_lin,   &
                             wfacpbl1, kpbl1, wfacpbl2, kpbl2,               &
                             l_hires_corr, l_restore_sfcinv, extrapol_dist,  &
                             l_sfc_inv, zextrap, slope, lacc)

    ! Atmospheric fields
    REAL(wp), INTENT(IN)  :: temp_in (:,:) ! input temperature field
    REAL(wp), INTENT(OUT) :: temp_out(:,:) ! output temperature field

    ! Coordinate fields
    REAL(wp), INTENT(IN) :: z3d_in(:,:)   ! 3D height coordinate field of input data
    REAL(wp), INTENT(IN) :: z3d_out(:,:)  ! 3D height coordinate field of output data

    ! Dimension parameters
    INTEGER , INTENT(IN) :: nblks      ! Number of blocks
    INTEGER , INTENT(IN) :: jcs, jce   ! Length of last block
    INTEGER , INTENT(IN) :: nlevs_in   ! Number of input levels
    INTEGER , INTENT(IN) :: nlevs_out  ! Number of output levels
    INTEGER , INTENT(IN) :: jb         ! index of current block

    ! Coefficients
    REAL(wp), INTENT(IN) :: coef1(:,:)    ! coefficient for linear term
    REAL(wp), INTENT(IN) :: coef2(:,:)    ! coefficient for quadratic term
    REAL(wp), INTENT(IN) :: coef3(:,:)    ! coefficient for cubic term
    REAL(wp), INTENT(IN) :: wfac_lin(:,:) ! weighting factor for linear interpolation

    INTEGER , INTENT(IN) :: idx0_cub(:,:) ! index of upper level, cubic interpolation
    INTEGER , INTENT(IN) :: idx0_lin(:,:) ! index of upper level, linear interpolation

    INTEGER , INTENT(IN) :: bot_idx_cub(:)! index of lowest level for which cubic interpolation is possible
    INTEGER , INTENT(IN) :: bot_idx_lin(:)! index of lowest level for which cubic interpolation is possible
    INTEGER , INTENT(IN) :: kpbl1(:)      ! index of model level immediately above (by default) 500 m AGL
    REAL(wp), INTENT(IN) :: wfacpbl1(:)   ! corresponding interpolation coefficient
    INTEGER , INTENT(IN) :: kpbl2(:)      ! index of model level immediately above (by default) 1000 m AGL
    REAL(wp), INTENT(IN) :: wfacpbl2(:)   ! corresponding interpolation coefficient

    REAL(wp), OPTIONAL, INTENT(IN) :: zextrap(:)    ! AGL height from which downward extrapolation starts (in postprocesing mode)

    ! Logical switch if slope-based reduction of surface inversion is to be performed
    ! (recommended when target model has a much finer resolution than source model)
    LOGICAL, INTENT(IN) :: l_hires_corr

    ! Logical switch if surface inversion is to be restored after downward extrapolation
    ! (may be set to .FALSE. when interpolating on constant height/pressure levels)
    LOGICAL, INTENT(IN) :: l_restore_sfcinv

    ! Logical switch if modified temperature with surface inversion removed below zpbl1
    LOGICAL, INTENT(IN) :: l_sfc_inv

    REAL(wp), INTENT(IN) :: extrapol_dist ! Maximum extrapolation distance using the local vertical gradient
    LOGICAL, INTENT(IN), OPTIONAL :: lacc ! If true, use openacc

    REAL(wp), INTENT(IN), OPTIONAL :: slope(:)  ! slope of mass points
    ! LOGICAL, OPTIONAL,  INTENT(IN) :: opt_lmask(:,:)

    ! LOCAL VARIABLES

    INTEGER  :: jk, jk1, jc, jk_start, jk_start_in, jk_start_out, ik1(nproma)
    REAL(wp) :: wfac, sfcinv

    REAL(wp), DIMENSION(nproma) :: temp1, temp2, vtgrad_up, zdiff_inout, &
                                   redinv1, redinv2, tmsl, tmsl_mod
    LOGICAL , DIMENSION(nproma) :: l_found
    REAL(wp), DIMENSION(nproma,nlevs_in)  :: zalml_in, sfc_inv, temp_mod, g1, g2, g3
    REAL(wp), DIMENSION(nproma,nlevs_in-1) :: zalml_in_d
    REAL(wp), DIMENSION(nproma,nlevs_out) :: zalml_out
    LOGICAL :: lzacc ! non-optional version of lacc
    CHARACTER(len=*), PARAMETER :: routine = 'temperature_intp'

!-------------------------------------------------------------------------

    CALL set_acc_host_or_device(lzacc, lacc)

    ! return, if nothing to do:
    IF ((nblks == 0) .OR. ((nblks == 1) .AND. ((jce-jcs+1) == 0))) RETURN

    IF (l_hires_corr .AND. .NOT. PRESENT(slope)) CALL finish("temperature_intp:",&
      "slope correction requires slope data as input")


!$OMP PARALLEL private(jc, jk, zalml_in, zalml_out)

    !$ACC DATA IF(lzacc) &
    !$ACC   PRESENT(num_lev) &
    !$ACC   CREATE(ik1, temp1, temp2, vtgrad_up, zdiff_inout) &
    !$ACC   CREATE(redinv1, redinv2, tmsl, tmsl_mod) &
    !$ACC   CREATE(l_found, zalml_in, sfc_inv, temp_mod, g1, g2, g3) &
    !$ACC   CREATE(zalml_out, zalml_in_d) &
    !$ACC   PRESENT(temp_in, temp_out, z3d_in, z3d_out, coef1, coef2, coef3, wfac_lin, idx0_cub, idx0_lin) &
    !$ACC   PRESENT(bot_idx_cub, bot_idx_lin, kpbl1, wfacpbl1, kpbl2, wfacpbl2, zextrap, slope)

    !$ACC PARALLEL ASYNC(1) DEFAULT(PRESENT) IF(lzacc)
    !$ACC LOOP SEQ
    DO jk = 1, nlevs_in
      !$ACC LOOP GANG(STATIC: 1) VECTOR
      DO jc = 1, nproma
        zalml_in(jc, jk) = (jc - 1) * nlevs_in + jk
      END DO
    END DO
    !$ACC LOOP SEQ
    DO jk = 1, nlevs_out
      !$ACC LOOP GANG(STATIC: 1) VECTOR
      DO jc = 1, nproma
        zalml_out(jc, jk) = (jc - 1) * nlevs_out + jk
      END DO
    END DO
    !$ACC END PARALLEL

!-------------------------------------------------------------------------

!$OMP DO PRIVATE(jk,jk1,jc,jk_start,jk_start_in,jk_start_out,ik1,wfac,sfcinv,    &
!$OMP            temp1,temp2,tmsl,tmsl_mod,vtgrad_up,zdiff_inout,redinv1,redinv2,l_found,&
!$OMP            zalml_in_d,temp_mod,sfc_inv,g1,g2,g3) ICON_OMP_DEFAULT_SCHEDULE

    IF (jb == nblks) THEN
      !$ACC PARALLEL LOOP GANG VECTOR DEFAULT(PRESENT) ASYNC(1) IF(lzacc)
      DO jc=jce+1, nproma
        temp_out(jc,:)  = -99.0_wp
      ENDDO
      !$ACC END PARALLEL LOOP
    ENDIF

    IF (itype_pres_msl >= 3) THEN ! l_pz_mode=.FALSE.
      !$ACC PARALLEL ASYNC(1) DEFAULT(PRESENT) IF(lzacc)
      !$ACC LOOP GANG(STATIC: 1) VECTOR
      DO jc = jcs, jce
        ! Temperature at height zpbl1
     
        temp1(jc) = wfacpbl1(jc) *temp_in(jc,kpbl1(jc)) + &
              (1._wp-wfacpbl1(jc))*temp_in(jc,kpbl1(jc)+1)

        ! Temperature at height zpbl2
        temp2(jc) = wfacpbl2(jc) *temp_in(jc,kpbl2(jc)) + &
              (1._wp-wfacpbl2(jc))*temp_in(jc,kpbl2(jc)+1)

        ! Vertical gradient between zpbl1 and zpbl2
        vtgrad_up(jc) = (temp2(jc) - temp1(jc))/(zpbl2 - zpbl1)

        ! Set reasonable limits
        ! vtgrad_up(jc) = MAX(vtgrad_up(jc),-8.5e-3_wp)
        ! vtgrad_up(jc) = MIN(vtgrad_up(jc),-1.5e-3_wp)

        ! height distance between lowest input and output grid point
        ! (negative if extrapolation takes place)
        zdiff_inout(jc) = z3d_out(jc,nlevs_out) - z3d_in(jc,nlevs_in)
      ENDDO

      IF (l_sfc_inv) THEN
        !$ACC LOOP SEQ
        DO jk1 = 1, nlevs_in
          !$ACC LOOP GANG(STATIC: 1) VECTOR
          DO jc = jcs, jce

            ! Height above lowest model level
            zalml_in(jc,jk1) = z3d_in(jc,jk1) - z3d_in(jc,nlevs_in)

            ! remove surface inversion from temp
            IF (zalml_in(jc,jk1) < zpbl1) THEN
              temp_mod(jc,jk1) = temp1(jc)+vtgrad_up(jc)*(zalml_in(jc,jk1)-zpbl1) 
            ELSE
              temp_mod(jc,jk1) = temp_in(jc,jk1)
            ENDIF

            ! "surface inversion", defined by the difference between the extrapolated
            ! temperature from above and the original input temperature
            !
            ! Note: the extrapolated temperature may also be slightly colder than the original
            ! one in the presence of (super-)adiabatic temperature gradients. In such
            ! cases, using vtgrad_up for downward extrapolation is still appropriate because
            ! the atmosphere in narrow mountain valleys usually does not become adiabatic
            ! because of slope heating
            !
            sfc_inv(jc,jk1) = temp_mod(jc,jk1) - temp_in(jc,jk1)

          ENDDO
        ENDDO
      !$ACC END PARALLEL

      jk_start_in = start_idx_threshold(zpbl1, zalml_in(:,:), jcs-jce+1, nlevs_in, lzacc)

      ELSE ! l_sfc_inv=.FALSE.

        temp_mod = temp_in

      END IF

    ENDIF

    ! Reduce the surface inversion strength depending on the slope of the target point and
    ! on the height difference between the source and target topography.
    !
    ! The following empirical modifications are designed for the case that the target
    ! model has a significantly finer spatial resolution than the source model,
    ! implying that nocturnal surface inversions should be removed on grid points lying
    ! higher than on the source grid (i.e. mountain ranges not resolved in the source model)
    ! and on slope points because cold air drainage in reality lets the cold air
    ! accumulate at the valley bottom.

    ! !$ACC PARALLEL DEFAULT(PRESENT) ASYNC(1) IF(lzacc)
    ! IF (l_hires_corr) THEN ! Not executed
    !   !$ACC LOOP GANG(STATIC: 1) VECTOR
    !   DO jc = jcs, jce

    !     IF (slope(jc) <= 5.e-2_wp) THEN ! 50 m/km
    !       redinv1(jc) = 1._wp
    !     ELSE IF (slope(jc) <= 3.5e-1_wp) THEN ! 350 m/km
    !       redinv1(jc) = 1._wp - LOG(slope(jc)/5.e-2_wp)/LOG(10._wp)
    !     ELSE
    !       redinv1(jc) = 1._wp - LOG(7._wp)/LOG(10._wp)
    !     ENDIF

    !     ! zdiff_inout > 0 means that the target grid point is higher than the source point
    !     IF (zdiff_inout(jc) <= 100._wp) THEN
    !       redinv2(jc) = 1._wp + MAX(0._wp,MIN(1.5_wp,-1.e-3_wp*zdiff_inout(jc)))
    !     ELSE IF (zdiff_inout(jc) <= 300._wp) THEN
    !       redinv2(jc) = 1._wp - (zdiff_inout(jc)-100._wp)/200._wp
    !     ELSE
    !       redinv2(jc) = 0._wp
    !     ENDIF
    !   ENDDO

    !   ! Reduce surface inversion if sfc_inv > 0 (i.e. there is really enhanced static stability)
    !   !$ACC LOOP SEQ
    !   DO jk1 = jk_start_in, nlevs_in
    !     !$ACC LOOP GANG(STATIC: 1) VECTOR
    !     DO jc = jcs, jce
    !       IF (sfc_inv(jc,jk1) > 0._wp) THEN
    !         sfc_inv(jc,jk1) = sfc_inv(jc,jk1)*MIN(1._wp, redinv1(jc)*redinv2(jc))
    !       ENDIF
    !     ENDDO
    !   ENDDO

    ! ENDIF

    !$ACC LOOP GANG(STATIC: 1) VECTOR
    DO jc = jcs, jce
      ! Compute vertical gradients of input data
      g1(jc,1) = (temp_mod(jc,1)-temp_mod(jc,2))/ &
            (z3d_in(jc,1)-z3d_in(jc,2))
      g1(jc,2) = (temp_mod(jc,2)-temp_mod(jc,3))/ &
            (z3d_in(jc,2)-z3d_in(jc,3))
      ! Compute vertical gradients of gradients
      g2(jc,2-1) = (g1(jc,1)-g1(jc,2))/(z3d_in(jc,1)-z3d_in(jc,3))
    ENDDO
    !$ACC LOOP SEQ
    DO jk1 = 3, nlevs_in-1
      !$ACC LOOP GANG(STATIC: 1) VECTOR
      DO jc = jcs, jce
        ! Compute vertical gradients of input data
        g1(jc,jk1) = (temp_mod(jc,jk1 )-temp_mod(jc,jk1+1))/ &
                      (z3d_in(jc,jk1)-z3d_in(jc,jk1+1))
        ! Compute vertical gradients of gradients
        g2(jc,jk1-1) = (g1(jc,jk1-1)-g1(jc,jk1))/(z3d_in(jc,jk1-1)-z3d_in(jc,jk1+1))
        ! Compute third-order vertical gradients
        g3(jc,jk1-2) = (g2(jc,jk1-2)-g2(jc,jk1-1))/(z3d_in(jc,jk1-2)-z3d_in(jc,jk1+1))
      ENDDO
    ENDDO

    ! Now perform vertical interpolation, based on the modified temperature field
    ! mode for interpolating initial data

    !$ACC LOOP SEQ
    DO jk = 1, nlevs_out
      !$ACC LOOP GANG(STATIC: 1) VECTOR PRIVATE(jk1)
      DO jc = jcs, jce
        IF (jk <= bot_idx_cub(jc)) THEN

          ! cubic interpolation
          jk1 = idx0_cub(jc,jk)
          temp_out(jc,jk) = temp_mod(jc,jk1)           + coef1(jc,jk)*g1(jc,jk1) + &
                                coef2(jc,jk)*g2(jc,jk1) + coef3(jc,jk)*g3(jc,jk1)

        ELSE IF (jk <= bot_idx_lin(jc)) THEN

          ! linear interpolation
          jk1 = idx0_lin(jc,jk)
          temp_out(jc,jk) = wfac_lin(jc,jk)*temp_mod(jc,jk1) + &
            (1._wp-wfac_lin(jc,jk))*temp_mod(jc,jk1+1)

        ELSE

          ! linear extrapolation using the temperature gradient in 500m-1000m 
          ! wfac_lin carries the extrapolation distance
          temp_out(jc,jk) = temp_mod(jc,nlevs_in) + wfac_lin(jc,jk)*vtgrad_up(jc) 

        ENDIF

        ! Height above lowest model level - needed for restoring the surface inversion
        zalml_out(jc,jk) = z3d_out(jc,jk) - z3d_out(jc,nlevs_out)

      ENDDO
    ENDDO

    !$ACC END PARALLEL

    ! Finally, subtract surface inversion from preliminary temperature field
    IF (l_sfc_inv) THEN
      IF (l_restore_sfcinv) THEN 

        jk_start_out = start_idx_threshold(zpbl1, zalml_out(:,:), jce-jcs+1, nlevs_out, lzacc)

        !$ACC PARALLEL DEFAULT(PRESENT) ASYNC(1) IF(lzacc)
        !$ACC LOOP GANG VECTOR COLLAPSE(2)
        DO jk = 1, nlevs_in-1
          DO jc = jcs, jce
            zalml_in_d(jc,jk) = 1.0_wp / (zalml_in(jc,jk) - zalml_in(jc,jk+1))
          ENDDO
        ENDDO
        !$ACC END PARALLEL

        jk_start = jk_start_in - 1
        DO jk = jk_start_out, nlevs_out
          !$ACC PARALLEL DEFAULT(PRESENT) ASYNC(1) IF(lzacc)
          !$ACC LOOP GANG(STATIC: 1) VECTOR
          DO jc = jcs, jce
            l_found(jc) = .FALSE.
          ENDDO
          !$ACC LOOP SEQ
          DO jk1 = jk_start, nlevs_in-1
            !$ACC LOOP GANG(STATIC: 1) VECTOR PRIVATE(wfac, sfcinv)
            DO jc = jcs, jce
              IF(.NOT. l_found(jc)) THEN
                IF (zalml_out(jc,jk) >= zpbl1) THEN
                  l_found(jc) = .TRUE.
                  ik1(jc)     = jk_start
                ELSE IF (zalml_out(jc,jk) <  zalml_in(jc,jk1) .AND. &
                          zalml_out(jc,jk) >= zalml_in(jc,jk1+1)) THEN

                  wfac = (zalml_out(jc,jk)-zalml_in(jc,jk1+1)) * zalml_in_d(jc,jk1)
                  sfcinv = wfac*sfc_inv(jc,jk1) + (1._wp-wfac)*sfc_inv(jc,jk1+1)

                  l_found(jc) = .TRUE.
                  ik1(jc)     = jk1

                  temp_out(jc,jk) =  temp_out(jc,jk) - sfcinv

                ELSE IF (zalml_out(jc,jk) > zalml_in(jc,jk_start)) THEN
                  l_found(jc) = .TRUE.
                  ik1(jc)     = jk_start
                ENDIF
              ENDIF
            ENDDO
#ifndef _OPENACC
            ! ACC: the following EXIT would be illegal within an OpenACC Kernel, thus we skip this CPU optimization
            IF (ALL(l_found(1:jce-jcs+1))) EXIT
#endif
          ENDDO
          !$ACC END PARALLEL
          jk_start = minval_1d(ik1(1:jce-jcs+1), lzacc)
        ENDDO
      ENDIF
    ENDIF

    !$ACC WAIT
    !$ACC END DATA
!$OMP END DO NOWAIT
!$OMP END PARALLEL

  END SUBROUTINE temperature_intp

END MODULE mo_art_mloz
