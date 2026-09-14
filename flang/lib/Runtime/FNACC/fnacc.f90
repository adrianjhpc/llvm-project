! Public FnACC device-selection interfaces. Compile once with the host compiler.
module fnacc
  use, intrinsic :: iso_c_binding, only: c_int32_t
  implicit none
  private
  public :: fnacc_get_num_devices, fnacc_set_device_num, fnacc_get_device_num
  interface
    function fnacc_get_num_devices() bind(C, name="fnacc_get_num_devices") result(n)
      import :: c_int32_t
      integer(c_int32_t) :: n
    end function
    subroutine fnacc_set_device_num(device) bind(C, name="fnacc_set_device_num")
      import :: c_int32_t
      integer(c_int32_t), value :: device
    end subroutine
    function fnacc_get_device_num() bind(C, name="fnacc_get_device_num") result(device)
      import :: c_int32_t
      integer(c_int32_t) :: device
    end function
  end interface
end module
