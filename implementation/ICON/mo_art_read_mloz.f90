!---------------------------------------------------------
!---Read machine learning coefs & ozone scalings from ncfiles for ozone parameterization
!---------------------------------------------------------
!
! mo_art_read_mloz
! This module provides a simple linear ozone chemistry
! first introduced by McLinden 2000
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

MODULE mo_art_read_mloz
  USE mo_art_chem_data,                 ONLY: t_art_mloz
  USE mo_kind,                          ONLY: wp
  USE mo_exception,                     ONLY: message, message_text
  USE mo_read_interface,                ONLY: openInputFile, closeFile, on_cells, &
                                            & read_3D, read_3D_extdim, t_stream_id
  USE mo_model_domain,                  ONLY: p_patch

  IMPLICIT NONE

  PRIVATE

  PUBLIC  :: art_mloz_coefs_read
  PUBLIC  :: art_mloz_deallocate

CONTAINS

SUBROUTINE art_mloz_coefs_read(art_mloz,ncoefs,nproma,nlev,nlev_ext,nblks,jg,l_add_feature)

!<
! SUBROUTINE art_mloz_read
! Routine reads in values needed for machine learning-based ozone parameterization
! based on Nowack et al. (2018)
! File 'machine_learning_coefs.nc', 'machine_learning_scalings.nc' & 'altitudes_external_model.nc'
! has to be linked in the output directory.
! Part of Module: mo_art_read_mloz
! Author: Yiling Ma, KIT
! Created on: 2024-08-12
!>

  IMPLICIT NONE 

  INTEGER, INTENT(in) :: &
    &  ncoefs,nproma,nlev,nlev_ext,nblks, & !< dimensions of arrays
    &  jg  ! domain id
  LOGICAL, INTENT(in) :: l_add_feature ! if add specific humidity/pressure as additional predictor

  TYPE(t_art_mloz), INTENT(inout)    :: &
    &  art_mloz                  !< Pointer to machine learning coef fields

  TYPE(t_stream_id) :: stream_id0,stream_id1,stream_id2,stream_id3,stream_id4,stream_id5,stream_id6


  CHARACTER*120 :: ml_meso_o3_file,ml_coef_file,ml_scaling_file1,ml_scaling_file2,ml_scaling_file3,altitude_external_file,ml_interc_file
  CHARACTER(len=*), PARAMETER :: routine = 'mo_art_read_mloz'

  ml_meso_o3_file = 'mesosphere_o3.nc'
  ml_coef_file = 'machine_learning_coefs.nc'
  ml_interc_file = 'machine_learning_intercepts.nc'
  ml_scaling_file1 = 'machine_learning_ozone_scalings.nc'
  ml_scaling_file2 = 'machine_learning_temp_scalings.nc'
  ml_scaling_file3 = 'machine_learning_additional_feature_scalings.nc'
  altitude_external_file = 'altitudes_external_model.nc'

  ! allocate mesosphere o3 climatologies
  IF (.NOT. ALLOCATED(art_mloz%ml_o3_meso)) THEN
    ALLOCATE(art_mloz%ml_o3_meso(nproma, nlev, nblks, 12))
    art_mloz%ml_o3_meso = 0._wp
  END IF

  ! allocate coefs
  IF (.NOT. ALLOCATED(art_mloz%ml_coefs)) THEN
    ALLOCATE(art_mloz%ml_coefs(nproma, nlev_ext, nblks, ncoefs))
    art_mloz%ml_coefs = 0._wp
  END IF
  ! allocate ozone&temp scalings
  IF (.NOT. ALLOCATED(art_mloz%ml_ozone_mean)) THEN
    IF (art_mloz%l_yscale_130lev) THEN
      ALLOCATE(art_mloz%ml_ozone_mean(nproma, nlev, nblks)) !(nproma, nlev, nblks) or (nproma, nlev_ext, nblks)
    ELSE
      ALLOCATE(art_mloz%ml_ozone_mean(nproma, nlev_ext, nblks))
    END IF
    art_mloz%ml_ozone_mean = 0._wp
  END IF
  IF (.NOT. ALLOCATED(art_mloz%ml_ozone_scale)) THEN
    IF (art_mloz%l_yscale_130lev) THEN
      ALLOCATE(art_mloz%ml_ozone_scale(nproma, nlev, nblks)) !(nproma, nlev, nblks) or (nproma, nlev_ext, nblks)
    ELSE
      ALLOCATE(art_mloz%ml_ozone_scale(nproma, nlev_ext, nblks))
    END IF
    art_mloz%ml_ozone_scale = 0._wp
  END IF
  IF (.NOT. ALLOCATED(art_mloz%ml_temp_mean)) THEN
    ALLOCATE(art_mloz%ml_temp_mean(nproma, nlev, nblks)) !(nproma,z,nblks)
    art_mloz%ml_temp_mean = 0._wp
  END IF
  IF (.NOT. ALLOCATED(art_mloz%ml_temp_scale)) THEN
    ALLOCATE(art_mloz%ml_temp_scale(nproma, nlev, nblks)) !(nproma,z,nblks)
    art_mloz%ml_temp_scale = 0._wp
  END IF
  IF (.NOT. ALLOCATED(art_mloz%z3d_ext)) THEN
    ALLOCATE(art_mloz%z3d_ext(nproma, art_mloz%n_zfile, nblks)) !(nproma,z,nblks)
    art_mloz%z3d_ext = 0._wp
  END IF  
  IF (l_add_feature) THEN
    IF (.NOT. ALLOCATED(art_mloz%ml_feature_mean)) THEN
      ALLOCATE(art_mloz%ml_feature_mean(nproma, nlev, nblks)) !(nproma,z,nblks)
      art_mloz%ml_feature_mean = 0._wp
    END IF
    IF (.NOT. ALLOCATED(art_mloz%ml_feature_scale)) THEN
      ALLOCATE(art_mloz%ml_feature_scale(nproma, nlev, nblks)) !(nproma,z,nblks)
      art_mloz%ml_feature_scale = 0._wp
    END IF
  END IF
  IF (art_mloz%l_intercept) THEN ! read machine learning intercepts from ncfile
    IF (.NOT. ALLOCATED(art_mloz%ml_intcs)) THEN
      ALLOCATE(art_mloz%ml_intcs(nproma, nlev_ext, nblks)) !(nproma,z,nblks)
      art_mloz%ml_intcs = 0._wp
    END IF
  END IF

  ! art_mloz%n_coefs = ncoefs

  ! read o3 climatologies from ncfile
  ! CALL openInputFile(stream_id=stream_id0, filename=ml_meso_o3_file, patch=p_patch(jg))
  ! ! read geometric height of ukesm
  ! CALL read_3D(stream_id=stream_id0, location=on_cells, variable_name='o3', fill_array=art_mloz%ml_o3_meso(:,:,:))
  ! CALL closeFile(stream_id0)
  
  ! read o3 monthly climatologies from ncfile
  CALL openInputFile(stream_id=stream_id0, filename=ml_meso_o3_file, patch=p_patch(jg))
  ! read geometric height of ukesm
  CALL read_3D_extdim(stream_id=stream_id0, location=on_cells, variable_name='o3', &
    & fill_array=art_mloz%ml_o3_meso(:,:,:,:),extdim_name="time",levelsDimName="height") 
  !(cell,heights,time) -> (nproma,heights,blks,time)
  WRITE(message_text,*) 'SHAPE(art_mloz%ml_o3_meso)=', SHAPE(art_mloz%ml_o3_meso)
  CALL message(TRIM(routine),TRIM(message_text))
  CALL closeFile(stream_id0)

  ! read machine learning coefs from ncfile
  CALL openInputFile(stream_id1, ml_coef_file, p_patch(jg)) !, p_patch(jg)
  CALL read_3D_extdim(stream_id=stream_id1, location=on_cells, variable_name='coefs', &
  & fill_array=art_mloz%ml_coefs(:,:,:,:),extdim_name="c",levelsDimName="z") !(cell,levels,c) -> (nproma,levels,blks,c)
  ! write (*,*), 'SHAPE(art_mloz%ml_coefs)=',SHAPE(art_mloz%ml_coefs)
  CALL closeFile(stream_id1)

  ! read geometric height of external model from ncfiles
  CALL openInputFile(stream_id=stream_id2, filename=altitude_external_file, patch=p_patch(jg))
  ! read geometric height of ukesm
  CALL read_3D(stream_id=stream_id2, location=on_cells, variable_name='altitude', fill_array=art_mloz%z3d_ext(:,:,:))
  ! write (*,*), 'SHAPE(art_mloz%z3d_ext)=',SHAPE(art_mloz%z3d_ext)
  ! write (*,*), 'AVERAGE(art_mloz%z3d_ext)=',SUM(art_mloz%z3d_ext)/SIZE(art_mloz%z3d_ext)
  CALL closeFile(stream_id2)

  ! read machine learning & ozone scalings from ncfiles
  CALL openInputFile(stream_id=stream_id3, filename=ml_scaling_file1, patch=p_patch(jg)) !, p_patch(jg)
  ! read mean of ozone
  CALL read_3D(stream_id=stream_id3, location=on_cells, variable_name='y_mean', fill_array=art_mloz%ml_ozone_mean(:,:,:))
  ! read ozone scaling
  CALL read_3D(stream_id3, on_cells, 'y_scale', art_mloz%ml_ozone_scale(:,:,:))
  ! write (*,*), 'SHAPE(art_mloz%ml_ozone_mean)=',SHAPE(art_mloz%ml_ozone_mean)
  CALL closeFile(stream_id3)

  CALL openInputFile(stream_id=stream_id4, filename=ml_scaling_file2, patch=p_patch(jg)) !, p_patch(jg)
  ! read mean of ozone
  CALL read_3D(stream_id=stream_id4, location=on_cells, variable_name='x_mean', fill_array=art_mloz%ml_temp_mean(:,:,:))
  ! read ozone scaling
  CALL read_3D(stream_id4, on_cells, 'x_scale', art_mloz%ml_temp_scale(:,:,:))
  ! write (*,*), 'SHAPE(art_mloz%ml_ozone_mean)=',SHAPE(art_mloz%ml_ozone_mean)
  CALL closeFile(stream_id4)

  IF(l_add_feature) THEN
    WRITE(message_text,*) 'Use feature ',art_mloz%feature_added, ' additionally for mloz'
    CALL message(TRIM(routine),TRIM(message_text))
    
    CALL openInputFile(stream_id=stream_id5, filename=ml_scaling_file3, patch=p_patch(jg)) !, p_patch(jg)
    ! read mean of ozone
    CALL read_3D(stream_id=stream_id5, location=on_cells, variable_name='x1_mean', fill_array=art_mloz%ml_feature_mean(:,:,:))
    ! read ozone scaling
    CALL read_3D(stream_id5, on_cells, 'x1_scale', art_mloz%ml_feature_scale(:,:,:))
    CALL closeFile(stream_id5)
  END IF

  IF (art_mloz%l_intercept) THEN
    CALL openInputFile(stream_id=stream_id6, filename=ml_interc_file, patch=p_patch(jg)) !, p_patch(jg)
    CALL read_3D(stream_id=stream_id6,location=on_cells,variable_name='intercepts',fill_array=art_mloz%ml_intcs(:,:,:))
    CALL closeFile(stream_id6)
  END IF

  art_mloz%is_init = .TRUE.

END SUBROUTINE art_mloz_coefs_read


SUBROUTINE art_mloz_deallocate(art_mloz)
!<
! SUBROUTINE art_mloz_deallocate(art_mloz)
! This routines deallocates all allocated arrays MLOZ
! Part of Module: mo_art_read_mloz
! Author: Yiling Ma, KIT
! Initial Release: 2024-09-12
!>
  TYPE(t_art_mloz), INTENT(inout)    :: &
    &  art_mloz                    !< Pointer to ART chem fields

  DEALLOCATE(art_mloz%ml_o3_meso)
  DEALLOCATE(art_mloz%ml_coefs)
  DEALLOCATE(art_mloz%ml_ozone_mean)
  DEALLOCATE(art_mloz%ml_ozone_scale)
  DEALLOCATE(art_mloz%ml_temp_mean)
  DEALLOCATE(art_mloz%ml_temp_scale)
  IF(ALLOCATED(art_mloz%ml_feature_mean)) DEALLOCATE(art_mloz%ml_feature_mean)
  IF(ALLOCATED(art_mloz%ml_feature_scale)) DEALLOCATE(art_mloz%ml_feature_scale)
  DEALLOCATE(art_mloz%z3d_ext)
  IF (ALLOCATED(art_mloz%ml_intcs)) DEALLOCATE(art_mloz%ml_intcs)

  art_mloz%is_init = .FALSE.

END SUBROUTINE art_mloz_deallocate

END MODULE mo_art_read_mloz
