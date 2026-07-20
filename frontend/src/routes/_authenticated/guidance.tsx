import React from 'react'
import { createFileRoute } from '@tanstack/react-router'

export const Route = createFileRoute('/_authenticated/guidance')({
  component: Placeholder,
})

function Placeholder() {
  return (
    <div className="p-8 text-center border-2 border-dashed border-border rounded-lg m-4 text-slate-400">
      Guidance chat history under construction.
    </div>
  )
}
