! RUN: %python %S/test_errors.py %s %flang_fc1

subroutine empty_data_directives()
  !ERROR: FNACC ENTER DATA requires at least one COPYIN or CREATE clause
  !$fnacc enter data

  !ERROR: FNACC EXIT DATA requires at least one COPYOUT or DELETE clause
  !$fnacc exit data
end subroutine

