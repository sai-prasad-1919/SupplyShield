import React, { useEffect } from 'react'
import { createFileRoute, Outlet, useNavigate } from '@tanstack/react-router'
import { AppShell } from '../components/AppShell'
import { isAuthenticated } from '../lib/api'

export const Route = createFileRoute('/_authenticated')({
  component: AuthenticatedLayout,
})

function AuthenticatedLayout() {
  const navigate = useNavigate()

  useEffect(() => {
    if (!isAuthenticated()) {
      navigate({ to: '/login', replace: true })
    }
  }, [navigate])

  if (!isAuthenticated()) return null

  return (
    <AppShell>
      <Outlet />
    </AppShell>
  )
}
