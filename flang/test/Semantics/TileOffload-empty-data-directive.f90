! RUN: %python %S/test_errors.py %s %flang_fc1

subroutine empty_data_directives()
  !ERROR: TileOffload ENTER DATA requires at least one COPYIN or CREATE clause
  !$tileoff enter data

  !ERROR: TileOffload EXIT DATA requires at least one COPYOUT or DELETE clause
  !$tileoff exit data
end subroutine

